import torch 
import build_geom_dataset
import os 
import sys
import numpy as np
from pathlib import Path
sys.path.append('/mnt/justin/structure_super_resolution')
from configs.datasets_config import geom_with_h
import plotly.graph_objects as go


def plotly_visualize(point_cloud, point_cloud2, title='Gaussian Kernel Convolution'):
    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=point_cloud[:,0],
        y=point_cloud[:,1],
        z=point_cloud[:,2],
        mode='markers',
        marker=dict(color='red'),
        name="PC1"
    ))

    fig.add_trace(go.Scatter3d(
        x=point_cloud2[:, 0],
        y=point_cloud2[:, 1],
        z=point_cloud2[:, 2],
        mode='markers',
        marker=dict(color='blue'),
        name='PC2'
    ))

    fig.update_layout(title=title,
                  scene=dict(
                      xaxis_title='X',
                      yaxis_title='Y',
                      zaxis_title='Z'
                  ))
    fig.show()

def gaussian_kernel(point_cloud, t, min_radius, max_radius):
    '''
    Return gaussian convolved version of coordinates x 

        point_cloud: (B,N,3)
        t: (B,)
    '''
    if len(point_cloud.shape) == 2:
        point_cloud = torch.unsqueeze(point_cloud, 0) #(B,N,3)

    radius = t*(max_radius - min_radius) + min_radius  #(B,)
    radius = radius.view(-1, 1, 1)

    distance_matrices = torch.cdist(point_cloud, point_cloud) #(B,N,N)

    distance_matrices_radii = distance_matrices / radius 

    weights = torch.exp(-1 * distance_matrices_radii)

    # normalize weights
    norm = weights.sum(dim=2, keepdim=True)
    weights /= norm 

    # get smoothed versions of point cloud 
    smoothed_point_clouds = weights @ point_cloud #(B,N,3)
    
    return smoothed_point_clouds 

def viz_kernel():
    data_file = 'data/geom/geom_drugs_10.npy'
    device = torch.device("cuda")
    dtype = torch.float32

    split_data = build_geom_dataset.load_split_data(data_file, val_proportion=0.1, test_proportion=0.1, filter_size=None)
    transform = build_geom_dataset.GeomDrugsTransform(geom_with_h, False, device, True)
    dataloaders = {}
    for key, data_list in zip(['train', 'val', 'test'], split_data):
        dataset = build_geom_dataset.GeomDrugsDataset(data_list, transform=transform)
        shuffle = (key == 'train') and not True 

        # Sequential dataloading disabled for now.
        dataloaders[key] = build_geom_dataset.GeomDrugsDataLoader(
            sequential=True, dataset=dataset, batch_size=32,
            shuffle=shuffle)
    train_data = split_data[0]
    first_point_cloud = train_data[1][:,1:]

    batch_point_cloud = torch.from_numpy(np.stack([train_data[i][:,1:] for i in range(1)]))
    t = torch.tensor([1])

    smoothed_point_cloud = gaussian_kernel(batch_point_cloud, t, 3,3)

    plotly_visualize(batch_point_cloud[0].detach().cpu().numpy(), smoothed_point_cloud[0].detach().cpu().numpy(), title=f'Gaussian kernel smoothed radius={3}')

    smoothed_point_cloud = gaussian_kernel(batch_point_cloud, t, 2,2)

    plotly_visualize(batch_point_cloud[0].detach().cpu().numpy(), smoothed_point_cloud[0].detach().cpu().numpy(), title=f'Gaussian kernel smoothed radius={2}')

    smoothed_point_cloud = gaussian_kernel(batch_point_cloud, t, 1,1)

    plotly_visualize(batch_point_cloud[0].detach().cpu().numpy(), smoothed_point_cloud[0].detach().cpu().numpy(), title=f'Gaussian kernel smoothed radius={1}')

    smoothed_point_cloud = gaussian_kernel(batch_point_cloud, t, 0.5,0.5)

    plotly_visualize(batch_point_cloud[0].detach().cpu().numpy(), smoothed_point_cloud[0].detach().cpu().numpy(), title=f'Gaussian kernel smoothed radius={0.5}')


if __name__ == "__main__":
    viz_kernel()