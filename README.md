# 点云处理工具集

程序说明：这是《产品建模技术》课程的大作业demo，该项目是一个基于 VTK、numpy、scipy 的点云处理工具集，包含多种常见的点云处理模块，包括点云降噪、点云配准、整体误差计算等。适用于点云数据的可视化、预处理、配准以及分析。

## 项目结构

```bash
ProjectRoot/
├── vtkVisualization.py      # 最基础的点云可视化工具模块
├── vtkSTLtoPointCloud.py    # 将STL文件转换为点云格式的模块
├── vtkDenoising.py          # 点云噪点可视化模块
├── vtkKDTree.py             # 使用KD树针对性去噪
├── vtkPCA.py                # 主成分分析（PCA）用于点云特征提取的模块
├── vtkICP.py                # 基于ICP（迭代最近点）的点云配准模块
├── vtkError.py              # 点云误差计算模块
└── vtkReconstruction.py     # 点云重建模块
```

## 安装依赖

本项目基于Python 3.9.5版本开发，运行以下命令安装所有依赖库：

```bash
pip install -r requirements.txt
```

项目中使用的主要第三方库及其版本如下：

```bash
matplotlib==3.9.4
numpy==2.0.2
numpy-stl==3.2.0
scikit-learn==1.6.0
scipy==1.13.1
vtk==9.4.1
```