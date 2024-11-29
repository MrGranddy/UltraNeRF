import argparse

import os
import pprint
from typing import Dict, Any, List, Optional

import numpy as np
import imageio
import matplotlib.pyplot as plt
import torch

import run_ultra_nerf_old as run_nerf_ultrasound
from load_us import load_us_data

class UltrasoundNerfProcessor:
    def __init__(self, base_dir: str, exp_name: str):
        """
        Initialize the Ultrasound NeRF Processor
        
        Args:
            base_dir (str): Base directory for logs
            exp_name (str): Experiment name
        """
        self.base_dir = base_dir
        self.exp_name = exp_name
        
        # Load configuration
        config_path = os.path.join(base_dir, exp_name, 'config.txt')
        print('Args:')
        with open(config_path, 'r') as config_file:
            print(config_file.read())
        
        # Parse arguments
        parser = run_nerf_ultrasound.config_parser()
        model_no = 'model_200000'
        self.args = parser.parse_args([
            '--config', config_path, 
            '--ft_path', os.path.join(base_dir, exp_name, f"{model_no}.npy")
        ])
        
        # Load data
        self.model_name = self.args.datadir.split("/")[-1]
        self.images, self.poses, self.i_test = load_us_data(self.args.datadir)
        
        # Prepare dimensions and parameters
        self.H, self.W = self.images[0].shape
        self.H, self.W = int(self.H), int(self.W)
        
        self.images = self.images.astype(np.float32)
        self.poses = self.poses.astype(np.float32)
        
        # Set up rendering parameters
        self.near = 0.
        self.far = self.args.probe_depth * 0.001
        
        # Create NeRF model
        _, self.render_kwargs_test, _, _, _ = run_nerf_ultrasound.create_nerf(self.args)
        self._prepare_render_kwargs()
        
        # Compute spatial parameters
        self.sw = self.args.probe_width * 0.001 / float(self.W)
        self.sh = self.args.probe_depth * 0.001 / float(self.H)
        
        # Set up output directories
        self._setup_output_directories()
    
    def _prepare_render_kwargs(self):
        """Prepare rendering configuration"""
        self.render_kwargs_test["args"] = self.args
        bds_dict = {
            'near': torch.tensor(self.near).float(),
            'far': torch.tensor(self.far).float()
        }
        self.render_kwargs_test.update(bds_dict)
        
        print('Render kwargs:')
        pprint.pprint(self.render_kwargs_test)
    
    def _setup_output_directories(self):
        """Create output directories for processing results"""
        map_number = 0
        self.output_dir = os.path.join(
            self.base_dir, 
            self.exp_name, 
            f"output_maps_{self.model_name}_{map_number}"
        )
        self.output_dir_params = os.path.join(self.output_dir, "params")
        self.output_dir_output = os.path.join(self.output_dir, "output")
        
        # Create directories
        os.makedirs(self.output_dir_params, exist_ok=True)
        os.makedirs(self.output_dir_output, exist_ok=True)
    
    def show_colorbar(self, image, name=None, cmap='rainbow', np_a=False):
        """
        Generate and save a colorbar visualization of an image
        
        Args:
            image: Input image
            name: Output file name
            cmap: Colormap to use
            np_a: Whether input is a numpy array
        
        Returns:
            Matplotlib image output
        """
        figure = plt.figure()
        if np_a:
            image_out = plt.imshow(image, cmap=cmap)
        else:
            image_out = plt.imshow(image.numpy(), cmap=cmap, vmin=0, vmax=1)
        
        plt.tick_params(
            top=False, bottom=False, left=False, right=False,
            labelleft=False, labelbottom=False
        )
        
        m = plt.cm.ScalarMappable(cmap=cmap)
        m.set_clim(0., 1.)
        plt.colorbar(m)
        
        if name:
            figure.savefig(name)
        plt.close(figure)
        
        return image_out
    
    def process(self, save_interval: int = 300):
        """
        Process ultrasound images using NeRF rendering
        
        Args:
            save_interval: Interval for saving intermediate results
        """
        rendering_params_save = None
        
        for i, c2w in enumerate(self.poses):
            print(f"Processing image {i}")
            
            # Render ultrasound image
            rendering_params = run_nerf_ultrasound.render_us(
                self.H, self.W, 
                self.sw, self.sh, 
                c2w=c2w[:3, :4], 
                **self.render_kwargs_test
            )
            
            # Save generated intensity map
            imageio.imwrite(
                os.path.join(self.output_dir_output, f"Generated {1000 + i}.png"),
                torch.transpose(rendering_params['intensity_map'], 0, 1).numpy().astype(np.uint8)
            )
            
            # Accumulate rendering parameters
            rendering_params_save = self._accumulate_rendering_params(
                rendering_params, 
                rendering_params_save
            )
            
            # Save periodically
            if i == save_interval or (i % save_interval == 0 and i != 0):
                self._save_rendering_params(rendering_params_save)
                rendering_params_save = None
        
        # Save final batch of parameters
        if rendering_params_save:
            self._save_rendering_params(rendering_params_save)
    
    def _accumulate_rendering_params(
        self, 
        rendering_params: Dict[str, Any], 
        rendering_params_save: Optional[Dict[str, List]] = None
    ) -> Dict[str, List]:
        """
        Accumulate rendering parameters
        
        Args:
            rendering_params: Current rendering parameters
            rendering_params_save: Accumulated parameters
        
        Returns:
            Updated accumulated parameters
        """
        if rendering_params_save is None:
            rendering_params_save = {
                key: [] for key in rendering_params
            }
        
        for key, value in rendering_params.items():
            transposed_value = torch.transpose(value, 0, 1)
            rendering_params_save[key].append(transposed_value.numpy())
            
            # Sanity check to ensure values are not constant
            if np.all(rendering_params_save[key][0] == value):
                raise ValueError(f"Constant values detected for key: {key}")
        
        return rendering_params_save
    
    def _save_rendering_params(self, rendering_params_save: Dict[str, List]):
        """
        Save accumulated rendering parameters
        
        Args:
            rendering_params_save: Accumulated parameters to save
        """
        for key, value in rendering_params_save.items():
            file_path = os.path.join(self.output_dir_params, f"{key}.npy")
            
            np_to_save = np.array(value)
            
            # Load existing data if file exists, otherwise save new data
            if os.path.exists(file_path):
                np_existing = np.load(file_path)
                np_to_save = np.concatenate((np_existing, np_to_save), axis=0)
            
            np.save(file_path, np_to_save)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--basedir', type=str, default='./logs')
    parser.add_argument('--expname', type=str, default='exp')
    args = parser.parse_args()
    
    processor = UltrasoundNerfProcessor(args.basedir, args.expname)
    processor.process()

if __name__ == "__main__":
    main()