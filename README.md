# Differential Surfel Rasterization

This is the rasterization engine for the paper "2D Gaussian Splatting for  Geometrically Accurate Radiance Fields". If you can make use of it in your own research, please be so kind to cite us.
# Install
```bash
git clone https://github.com/hjh530/2DGS-AbsGS-diff-gaussian-rasterization.git
cd diff-gaussian-rasterization
pip install . --no-cache-dir
```
# Usage
## In gaussian_renderer/init.py：
```python
def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
  
    # screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    screenspace_points = torch.zeros((pc.get_xyz.shape[0],4), dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass
	...
```
## In gaussian_model.py,you need add a parameter like this：
```python
self.xyz_gradient_accum = torch.empty(0)
self.xyz_gradient_accumabs = torch.empty(0)
```
## In gaussian_model.py：
```python
def densify_and_prune(self, max_grad,max_gradabs, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        gradsabs = self.xyz_gradient_accumabs / self.denom
        gradsabs[gradsabs.isnan()] = 0.0


        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(gradsabs, max_gradabs, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

def add_densification_stats(self, viewspace_point_tensor, update_filter):
    # self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter], dim=-1, keepdim=True)
    self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1,
                                                         keepdim=True)
    self.xyz_gradient_accum_abs[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,2:], dim=-1,
                                                             keepdim=True)
    self.denom[update_filter] += 1
```
## In arguments/init.py,add a parameter like this:
```python
self.densify_grad_threshold = 0.0002
self.densify_gradabs_threshold = 0.0008
```
## In train.py:
```python
 if iteration < opt.densify_until_iter:
    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
         size_threshold = 20 if iteration > opt.opacity_reset_interval else None
         gaussians.densify_and_prune(opt.densify_grad_threshold,opt.densify_gradabs_threshold,opt.opacity_cull, scene.cameras_extent, size_threshold)
                
    if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
         gaussians.reset_opacity()           
```
# Citation
```bibtex
@misc{ye2024absgs,
    title={AbsGS: Recovering Fine Details for 3D Gaussian Splatting}, 
    author={Zongxin Ye and Wenyu Li and Sidun Liu and Peng Qiao and Yong Dou},
    year={2024},
    eprint={2404.10484},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}

@inproceedings{Huang2DGS2024,
    title={2D Gaussian Splatting for Geometrically Accurate Radiance Fields},
    author={Huang, Binbin and Yu, Zehao and Chen, Anpei and Geiger, Andreas and Gao, Shenghua},
    publisher = {Association for Computing Machinery},
    booktitle = {SIGGRAPH 2024 Conference Papers},
    year      = {2024},
    doi       = {10.1145/3641519.3657428}
}

@Article{kerbl3Dgaussians,
      author       = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
      title        = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
      journal      = {ACM Transactions on Graphics},
      number       = {4},
      volume       = {42},
      month        = {July},
      year         = {2023},
      url          = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}

