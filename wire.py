import numpy as np
import laspy, os
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import RANSACRegressor, LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


class Wire:
    """
    A Wire of conductor points
    """
    
    def __enter__(self):
        return self
    
    def __init__(self, span_name, span_dir, out_dir, min_points=65):
        self.span_laz=os.path.join(span_dir, span_name)
        self.out_dir=out_dir
        self.span_name=span_name
        self.min_points=min_points

    # Plot different views of a span
    def plot_span(self, all_data):
        X = all_data[:,[0]]
        Y = all_data[:,[1]]
        Z = all_data[:,[2]]
    
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

        ax1.scatter(X, Y)
        ax1.set_xlabel('x',fontsize=16)
        ax1.set_ylabel('y',fontsize=16)
        ax1.set_title('Top View',fontsize=20)
        ax1.tick_params(axis='both', which='major', labelsize=14)
    
        ax2.scatter(X, Z)
        ax2.set_xlabel('x',fontsize=16)
        ax2.set_ylabel('z',fontsize=16)
        ax2.set_title('Front View',fontsize=20)
        ax2.tick_params(axis='both', which='major', labelsize=14)
    
        ax3.scatter(Y, Z)
        ax3.set_xlabel('y',fontsize=16)
        ax3.set_ylabel('z',fontsize=16)
        ax3.set_title('Side View',fontsize=20)
        ax3.tick_params(axis='both', which='major', labelsize=14)
    
        plt.show()       

    # Fit parabola using RANSAC
    #def ransac(self, model,x,y):
        # residual_threshold is the most importent param that affect the fitted model
        #model = make_pipeline(model, RANSACRegressor(min_samples=3,max_trials=0.01,residual_threshold=1,random_state=42))
        #model.fit(x, y)
    
        #return model

    def fit_line(self, all_data): 
        X = all_data[:,[0]]
        Y = all_data[:,[1]]
        Z = all_data[:,[2]]

        # RANSAC + line fitting from top view
        models = []
        while len(Y) > self.min_points:  
            model = RANSACRegressor(LinearRegression(), min_samples=3,max_trials=0.01,residual_threshold=0.01,random_state=42)
            model.fit(X,Y)
            model_info = {}
            model_id = len(models)
        
            print('Model ', model_id)
            print('n_trials_: ', model.n_trials_)
            print('#inlier: ', sum(model.inlier_mask_))
            print('n_skips_no_inliers_: ', model.n_skips_no_inliers_)
            print('n_skips_invalid_data_: ', model.n_skips_invalid_data_)
            print('n_skips_invalid_model_: ', model.n_skips_invalid_model_)
            model_info['Y_inlier'] = np.array(Y[model.inlier_mask_])
            model_info['X_inlier'] = np.array(X[model.inlier_mask_])
            model_info['X_plot'] = np.arange(X.min(), X.max(), 0.1)[:, np.newaxis]
            model_info['Y_plot'] = model.predict(model_info['X_plot'])
            model_info['Y_outlier'] = np.array(Y[~model.inlier_mask_])
            model_info['X_outlier'] = np.array(X[~model.inlier_mask_])
            model_info['inlier_mask'] = model.inlier_mask_
            model_info['Z_inlier'] = np.array(Z[model.inlier_mask_])
            models.append(model_info)
        
            # Remove inliers from data
            Y = model_info['Y_outlier']
            X = model_info['X_outlier']
            Z = np.array(Z[~model.inlier_mask_])
    
        return models
    
    def fit_parabola(self, Y, Z): 
        # RANSAC + parabola from side view
        models = []
        while len(Y) > self.min_points:
            model = make_pipeline(PolynomialFeatures(2), RANSACRegressor(min_samples=3,max_trials=10,residual_threshold=0.01,random_state=42))
            model.fit(Y, Z)
            model_info = {}
            model_id = len(models)
        
            print('Model ', model_id)
            print('n_trials_: ', model['ransacregressor'].n_trials_)
            print('#inlier: ', sum(model['ransacregressor'].inlier_mask_))
            print('n_skips_no_inliers_: ', model['ransacregressor'].n_skips_no_inliers_)
            print('n_skips_invalid_data_: ', model['ransacregressor'].n_skips_invalid_data_)
            print('n_skips_invalid_model_: ', model['ransacregressor'].n_skips_invalid_model_)
            model_info['Y_inlier'] = np.array(Y[model['ransacregressor'].inlier_mask_])
            model_info['Z_inlier'] = np.array(Z[model['ransacregressor'].inlier_mask_])
            model_info['Y_plot'] = np.arange(Y.min(), Y.max(), 0.1)[:, np.newaxis]
            model_info['Z_plot'] = model.predict(model_info['Y_plot'])
            model_info['Y_outlier'] = np.array(Y[~model['ransacregressor'].inlier_mask_])
            model_info['Z_outlier'] = np.array(Z[~model['ransacregressor'].inlier_mask_])
            model_info['inlier_mask'] = model['ransacregressor'].inlier_mask_
            models.append(model_info)
        
            # Remove inliers from data
            Y = model_info['Y_outlier']
            Z = model_info['Z_outlier']
    
        return models
    
    def plot_models(self, models, model_type):        
        # Plot fitted model
        num_models = len(models)
        print(f'{num_models} {model_type}')
        fig, axes = plt.subplots(1, num_models, squeeze=False)
    
    
        if model_type == 'line':
            for i in range(num_models):
                #print(i)
                axes[0,i].scatter(models[i]['X_outlier'], models[i]['Y_outlier'], color='lightgray', label='Outliers')
                axes[0,i].scatter(models[i]['X_inlier'], models[i]['Y_inlier'], color='blue', label='Inliers')
                axes[0,i].set_title(f'Model{i}',fontsize=20)
                axes[0,i].set_xlabel('x',fontsize=16)
                axes[0,i].set_ylabel('y',fontsize=16)
                axes[0,i].plot(models[i]['X_plot'], models[i]['Y_plot'], color='r', linestyle='-',
                 linewidth=3, label='fitted line')
                axes[0,i].tick_params(axis='both', which='major', labelsize=14)
                axes[0,i].legend(fontsize=14) 
                 
        if model_type == 'parabola':
            for i in range(num_models):
                axes[0,i].scatter(models[i]['Y_outlier'], models[i]['Z_outlier'], color='lightgray', label='Outliers')
                axes[0,i].scatter(models[i]['Y_inlier'], models[i]['Z_inlier'], color='blue', label='Inliers')
                axes[0,i].set_title(f'Model{i}',fontsize=20)
                axes[0,i].set_xlabel('y',fontsize=16)
                axes[0,i].set_ylabel('z',fontsize=16)
                axes[0,i].plot(models[i]['Y_plot'], models[i]['Z_plot'], color='r', linestyle='-',
                 linewidth=3, label='fitted parabola')
                axes[0,i].tick_params(axis='both', which='major', labelsize=14)
                axes[0,i].legend(fontsize=14)

        plt.show()        
        
    def extract_wires(self):
        inFile = laspy.file.File(self.span_laz, mode = "r")
        dataset = np.vstack([inFile.X, inFile.Y, inFile.Z]).transpose()
        print('#point: ', len(dataset))
    
        # Rescale data
        all_data = MinMaxScaler().fit_transform(dataset)
    
        # Fit line from top view
        models = self.fit_line(all_data)     # [{line0}, {line1}, ...]
    
        wires = []
        line_masks = []
        # For each fitted line, fit parabola from side view
        for line in models:
            line['parabolas'] = self.fit_parabola(line['Y_inlier'], line['Z_inlier'])
            line_masks.append(line['inlier_mask'])
        
            parabola_masks = []
            # Extract individual wires with original coords 
            for parabola in line['parabolas']:
                parabola_masks.append(parabola['inlier_mask'])
                #print('#parabola: ', len(parabola_masks))
            
                wire = dataset
                for line_id in range(len(line_masks)):
                    if line_id == len(line_masks) - 1:
                        wire = dataset[np.array(line_masks[line_id]),:]
                        #print('#remaining_point: ', wire.shape[0])
                    else:
                        wire = dataset[~np.array(line_masks[line_id]),:]
                        #print('#remaining_point: ', wire.shape[0])
                for parabola_id in range(len(parabola_masks)):
                    if parabola_id == len(parabola_masks) - 1:
                        wire = wire[np.array(parabola_masks[parabola_id]),:]
                        wires.append(wire)
                        print(f'Wire{len(wires)}: {len(wire)} points')
                    else:
                        wire = wire[~np.array(parabola_masks[parabola_id]),:]
                        #print('#remaining_point: ', wire.shape[0])
                
        print(f'#points in {len(wires)} wires: {[len(wire) for wire in wires]}')
    
        # Save wires to file
        os.makedirs(self.out_dir, exist_ok=True)
        for id in range(len(wires)):
            np.savetxt(os.path.join(self.out_dir, f'{self.span_name[:-4]}_wire{id}.txt'), wires[id])
            
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
        
 
