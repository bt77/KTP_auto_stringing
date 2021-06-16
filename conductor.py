"""
Copied from Ben Thomas' catenary_best_fit: https://bitbucket.trimble.tools/projects/NMGROUP/repos/catenary_best_fit/browse?at=refs%2Fheads%2Fdev
"""


import numpy as np
from scipy.optimize import least_squares
from scipy.optimize import OptimizeResult
from scipy.linalg import lstsq
from scipy.spatial.transform import Rotation as R
from sklearn.metrics import mean_squared_error

class Conductor(object):

    def __init__(self):
        self.a_gen = 0.0
        self.a0 = 0.0
        self.a = 0.0
        self.min_gen = np.zeros((1, 3))
        self.min_obs = np.zeros((1, 3))
        self.min0 = np.zeros((1, 3))
        self.min_pred = np.zeros((1, 3))
        self.theta_gen = 0.0
        self.theta = 0.0
        self.fit_type = 'Catenary'
        self.length = 0.0
        self.start = np.zeros((1, 3))
        self.end = np.zeros((1, 3))
        self.gradient = 0.0
        self.intercept = 0.0
        self.success = None
        self.message = None
        self.catenary = Catenary()

    # generate catenary data for testing
    def random(self,
               lmin=10,
               lmax=500,
               emin=0,
               emax=1e6,
               nmin=0,
               nmax=1e6,
               hmin=0,
               hmax=1e3,
               num=0,
               noise=0.0,
               theta=None,
               seed=None):

        # create a random number generator
        rng = np.random.Generator(
            np.random.PCG64(seed)
        )

        # theta
        if theta is None:
            theta = np.pi * rng.uniform(-1, 1, 1)[0]
        self.theta_gen = theta
        self.theta = theta

        # length
        #self.length = rng.integers(lmin, lmax, 1)[0]
        self.length = lmin + (rng.beta(a=1.5, b=5) * (lmax-lmin))

        # start
        xs = rng.uniform(emin, emax, 1)[0]
        ys = rng.uniform(nmin, nmax, 1)[0]
        self.start = np.array([[xs, ys, 0]])

        # end
        xe = xs + (self.length * np.cos(-theta))
        ye = ys + (self.length * np.sin(-theta))
        self.end = np.array([[xe, ye, 0]])

        # minimum
        offset = (
                (np.floor(self.length/4) * abs(rng.standard_normal()))
                - np.floor(self.length/2)
        )
        self.min_gen = np.array(
            [[xs - (offset * np.cos(-theta)),
              ys - (offset * np.sin(-theta)),
              rng.uniform(hmin, hmax, 1)[0]]]
        )
        self.min_pred = self.min_gen
        self.min0 = self.min_gen
        self.min_obs = self.min_gen

        # a
        #self.a_gen = self.length * rng.uniform(1, 10, 1)[0]
        # empirical estimate of a from length based on UKPN data
        self.a_gen = (
                1 + (2500.000*self.length / 219.776 + self.length)
        ) * rng.uniform(0.5, 1.5, 1)[0]
        self.a = self.a_gen
        self.fit_type = True
        if num < 2:
            num = max(3, int(np.floor(self.length)))

        # generate
        data_gen = self.generate(num)

        # add some noise
        data_gen += rng.standard_normal(data_gen.shape) * noise

        return data_gen

    # generate a catenary after fitting using the start point and an end
    # point identified from the last predict
    # use num points or unit value based on length
    def generate(self, num=0):

        if num < 2:
            num = max(2, int(np.floor(self.length)))
        x = np.linspace(self.start[0, 0], self.end[0, 0], num=num)
        y = np.linspace(self.start[0, 1], self.end[0, 1], num=num)
        z = np.zeros(y.shape)
        X = np.array([x, y, z]).T

        return self.predict(X)

    # generate the best fit conductor from the modelled parameters
    def fit(self, X):

        bv = self.bounding_vector(X)
        self.length = np.sqrt(bv[0] ** 2 + bv[1] ** 2)
        self.gradient, self.intercept, self.theta = self.fit_2d(X)
        self.min_obs = self.get_min_obs(X)
        self.min0 = self.project_min0(
            X, self.min_obs, self.gradient, self.intercept
        )
        X_prime = self.transform_to_prime(X, self.min0, self.theta)

        number_of_points = len(X)
        if number_of_points > 2:
            self.fit_type = 'Catenary'
            self.a0 = (
                    1 + (2500.000 * self.length / 219.776 + self.length)
            )
            self.fit_catenary(X_prime[:, 0], X_prime[:, 2])
            return
        else:
            if number_of_points == 2:
                self.fit_type = 'Linear'
                self.fit_linear(X_prime[:, 0], X_prime[:, 2])
                return

        expression = 'Insufficient data points'
        message = 'X has length: {}, '.format(number_of_points)
        message += 'at least 2 points are required to model a conductor'
        raise InputError(expression, message)

    # generate the best fit conductor from the modelled parameters
    def fit_catenary(self, X, y):

        number_of_points = len(X)
        if number_of_points < 3:
            expression = 'Insufficient data points'
            message = 'X has length: {}, at '.format(number_of_points)
            message += 'least 3 points are required to model a catenary'
            raise InputError(expression, message)

        self.catenary.fit(X, y)

        min_pred_prime = (
            np.array([[
                self.catenary.model.x[1],
                0,
                self.catenary.model.x[2]]]
            )
        )
        self.min_pred = self.transform_from_prime(
            min_pred_prime, self.min0, self.theta
        )
        self.success = self.catenary.model.success
        self.message = self.catenary.model.message
        self.a = self.catenary.model.x[0]

    # fit a straight line to the data
    # required in some instances
    def fit_linear(self, X, y):

        if len(X) < 2:
            expression = 'Insufficient data points'
            number_of_points = len(X)
            if number_of_points < 3:
                expression = 'Insufficient data points'
                message = 'X has length: {}'.format(number_of_points)
                message += ', at least least 2 points'
                message += ' are required to model a straight line'
                raise InputError(expression, message)

        # use ols to fit in 2d (for z this time)
        M = np.array([np.ones(X.shape), X]).T  # design matrix
        p, res, rnk, s = lstsq(M, y)  # ols adjustment
        self.intercept = p[0]  # intercept
        self.gradient = p[1]  # gradient

        self.success = True
        self.message = 'linear model'
        self.min_pred = self.min0

    # using the model generated by fit_catenary() / fit_linear()
    # calculate catenary heights given a set of input data
    def predict(self, X):

        X_prime = self.transform_to_prime(X, self.min0, self.theta)
        min_pred_prime = self.transform_to_prime(
            self.min_pred, self.min0, self.theta
        )
        x = X_prime[:, 0]
        if self.fit_type == 'Catenary':
            z = self.catenary.predict(x)
        else:
            z = (self.gradient * x) + self.intercept
        y = np.zeros(z.shape)
        X_pred_prime = np.array([x, y, z]).T

        X_pred = self.transform_from_prime(
            X_pred_prime, self.min0, self.theta
        )
        self.start = self.transform_from_prime(
            X_pred_prime[np.argmin(x), :], self.min0, self.theta
        )
        self.end = self.transform_from_prime(
            X_pred_prime[np.argmax(x), :], self.min0, self.theta
        )

        return X_pred

    # return the maximum deviation of the conductor from a straight line
    def sag(self):

        start_prime = self.transform_to_prime(
            self.start, self.min_pred, self.theta
        )[0]
        end_prime = self.transform_to_prime(
            self.end, self.min_pred, self.theta
        )[0]
        min_pred_prime = self.transform_to_prime(
            self.min_pred, self.min_pred, self.theta
        )[0]
        p1 = np.array([start_prime[0], start_prime[2]])
        p2 = np.array([end_prime[0], end_prime[2]])
        p3 = np.array([min_pred_prime[0], min_pred_prime[2]])

        sag = np.cross(p2 - p1, p3 - p1) / np.linalg.norm(p2 - p1)
        return sag

    # use the lowest point as the observed conductor minimum point
    @staticmethod
    def get_min_obs(X):

        idx_minz = np.argmin(X, axis=0)[2]
        return X[idx_minz, :][:, np.newaxis].T

    # calculate min0 by projecting the observed minimum onto the best
    # fit 2D to ensure the translation to prime is centred on 0 and
    # to use as initial estimate for catenary best fit search
    # if transposed index will flip the axis for the projection
    # but flip back to generate min0
    def project_min0(self, X, min_obs, m, c):

        transpose, idx_col = self.transpose(X)

        A = np.array([0, c, min_obs[0, 2]])
        B = np.array([1, m + c, min_obs[0, 2]])
        P = min_obs[:, idx_col][0, :]
        AB = B - A
        AP = P - A
        R = A + (AB * (np.dot(AP, AB) / np.dot(AB, AB)))
        min0 = R[:, np.newaxis].T

        return min0[:, idx_col]

    # return true if the conductor is over the +/-45 degree threshold
    # used to transpose xy for least squares fit in 2D
    # return the transposed (or not) column index
    def transpose(self, X):

        lv = self.bounding_vector(X)
        transpose = (lv[0] < lv[1])
        if transpose:
            idx_col = [1, 0, 2]
        else:
            idx_col = [0, 1, 2]

        return transpose, idx_col

    # fit the data in 2D using ols
    def fit_2d(self, X):

        # avoid singularities in the ols adjustment by flipping the xy
        # axis when the orientation of the conductor is closer to North
        # than East. Do this by reindexing the columns with a list
        transpose, idx_col = self.transpose(X)

        x = X[:, idx_col[0]]
        y = X[:, idx_col[1]]

        M = np.array([np.ones(x.shape), x]).T  # design matrix
        p, res, rnk, s = lstsq(M, y)  # ols adjustment
        c = p[0]  # intercept
        m = p[1]  # gradient

        # calculate the angle of the conductor in 2D to
        # deal with the quadrants for transposed data
        if transpose:
            if m < 0:
                theta = np.arctan(m) + np.pi / 2
            else:
                theta = np.arctan(m) - np.pi / 2
        else:
            theta = -np.arctan(m)

        return m, c, theta

    # transform the catenary points into model space (prime coordinates)
    # translate data so that the point with the min z is at the origin
    # rotate the data so that the conductor points align with the x axis
    def transform_to_prime(self, X, translation, rotation):

        M = self.rotation_matrix(rotation)
        X_prime = R.from_matrix(M).apply(X - translation)

        return X_prime

    # reverse the transformation from prime coords back into real coords
    def transform_from_prime(self, X_prime, translation, rotation):

        M = self.rotation_matrix(-rotation)
        X = R.from_matrix(M).apply(X_prime) + translation

        return X

    # calculate the residuals from the observed data and modelled params
    # return the root of the square of the residuals
    def catenary_resid(self, x, e, h):
        return (self.catenary_z(e, x[0], x[1], x[2]) - h) ** 2

    # get the vector of the diagonal of the 2D box bounding the data
    @staticmethod
    def bounding_vector(X):

        idx_minx = np.argmin(X, axis=0)[0]
        idx_miny = np.argmin(X, axis=0)[1]
        idx_maxx = np.argmax(X, axis=0)[0]
        idx_maxy = np.argmax(X, axis=0)[1]
        x_diff = X[idx_maxx, 0] - X[idx_minx, 0]
        y_diff = X[idx_maxy, 1] - X[idx_miny, 1]

        return (x_diff, y_diff)

    # generate a rotation matrix given the angle of rotation in radians
    @staticmethod
    def rotation_matrix(theta):
        return [[np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]]

    # calculate the height of the catenary
    # given a set of x coordinates and params for the catenary equation
    @staticmethod
    def catenary_z(X, a, xmin, zmin):
        return zmin + (a * (np.cosh((X - xmin) / a) - 1))


class Catenary(object):

    def __init__(self):
        self.model = OptimizeResult()

    # generate a catenary after fitting using the start point and an end
    # point identified from the last predict
    # use num points or unit value based on length
    #def generate(self, num=0):

    #    if num < 2:
    #        num = max(2, int(np.floor(self.length)))
    #    x = np.linspace(self.start[0], self.end[0], num=num)
    #    y = np.linspace(self.start[1], self.end[1], num=num)
    #    z = np.zeros(y.shape)
    #    X = np.array([x, y, z]).T

    #    return self.predict(X)

    # generate the best fit conductor from the modelled parameters
    def fit(self, X, y):

        # use non-linear least squares to solve for the catenary params
        # empirical estimate of a from length based on UKPN data
        length = max(X) - min(X)
        a = 1 + (2500.000 * length / 219.776 + length)
        self.model = least_squares(
            self.residuals,     # func. for residuals each iteration
            np.array([a, min(X), min(y)]),  # array of params
            method='lm',    # Levenberg-Marquardt
            x_scale='jac',  # scale the params from the jacobian matrix
            args=(X, y)
        )

    # calculate the residuals from the observed data and modelled params
    # return the squared error
    def residuals(self, x, X, y):

        self.set_model(x)
        return (self.predict(X) - y)**2

    # using the model generated by fit_catenary() / fit_linear()
    # calculate catenary heights given a set of input data
    def predict(self, X):

        a = self.model.x[0]
        xmin = self.model.x[1]
        zmin = self.model.x[2]
        return zmin + (a * (np.cosh((X - xmin) / a) - 1))

    def score(self, X, y):
        return mean_squared_error(y, self.predict(X))

    def get_model(self):
        return self.model.x

    def set_model(self, x):
        self.model.x = x

class Error(Exception):
    """Base class for exceptions in this module."""
    pass


class InputError(Error):
    """Exception raised for errors in the input.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, expression, message):
        self.expression = expression
        self.message = message
