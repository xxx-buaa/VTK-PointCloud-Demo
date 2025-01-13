import numpy as np
import vtk
from vtkmodules.util.numpy_support import numpy_to_vtk
from scipy.spatial.transform import Rotation as R
from scipy.spatial import KDTree

# 渲染配置
render_source = True # 渲染实际加工点云
render_model = False # 渲染理论模型点云

def align_point_cloud(source, target):
    """
    使用PCA对源点云进行旋转和平移，使其与目标点云粗配准。
    返回配准后的源点云、旋转矩阵和平移向量。
    """
    # 计算PCA
    src_eigvec, src_centroid = compute_pca(source)
    tgt_eigvec, tgt_centroid = compute_pca(target)
    
    # 计算旋转矩阵
    rotation_matrix = tgt_eigvec @ src_eigvec.T

    # 确保旋转矩阵是正交的
    if np.linalg.det(rotation_matrix) < 0:
        rotation_matrix[:, -1] *= -1

    # 计算平移向量（单一向量）
    translation = tgt_centroid - (src_centroid @ rotation_matrix.T)

    # 旋转和翻译源点云
    aligned = (source @ rotation_matrix.T) + translation

    return aligned, rotation_matrix, translation

def compute_pca(point_cloud):
    """
    计算点云的PCA，返回旋转矩阵和质心。
    """
    centroid = np.mean(point_cloud, axis=0)
    centered = point_cloud - centroid
    cov = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # 排序特征值和特征向量
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    return eigenvectors, centroid

def vtk_visualize(source, target, source_color=(0.122, 0.467, 0.706), target_color=(0.839, 0.153, 0.157)):
    """
    使用VTK可视化两个点云。
    source: 实际加工点云
    target: 理论点云模型
    source_color: 实际加工点云颜色（默认为红色）
    target_color: 理论点云模型颜色（默认为蓝色）
    """
    def create_vtk_actor(point_cloud, color):
        vtk_points = vtk.vtkPoints()
        for point in point_cloud:
            vtk_points.InsertNextPoint(point)

        vertices = vtk.vtkCellArray()
        for i in range(vtk_points.GetNumberOfPoints()):
            vertices.InsertNextCell(1)
            vertices.InsertCellPoint(i)

        polydata = vtk.vtkPolyData()
        polydata.SetPoints(vtk_points)
        polydata.SetVerts(vertices)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetPointSize(2)
        actor.GetProperty().SetColor(color)
        return actor

    source_actor = create_vtk_actor(source, source_color)
    target_actor = create_vtk_actor(target, target_color)

    # 创建渲染器、渲染窗口和交互器
    renderer = vtk.vtkRenderer()
    # renderer.GradientBackgroundOn()
    renderer.AddActor(source_actor)
    renderer.AddActor(target_actor)
    # renderer.SetBackground(0.7, 0.85, 1.0)   # 起始颜色
    # renderer.SetBackground2(0.5, 0.75, 1.0)  # 结束颜色
    renderer.SetBackground(1, 1, 1)

    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(800, 600)

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)

    # 设置交互样式（可选）
    interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

    # 启动可视化
    interactor.Initialize()
    render_window.Render()
    interactor.Start()

if __name__ == '__main__':

    # 文件路径
    source_point_cloud_path = "src\\CalibrationPointCloud_denoised.txt"
    model_point_cloud_path = "src\\point_cloud.txt"

    theta = np.radians(25)  # 旋转角度（弧度）
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array([
        [cos_theta, -sin_theta, 0],
        [sin_theta,  cos_theta, 0],
        [0, 0, 1]
    ])

    source_pc = (np.loadtxt(source_point_cloud_path) @ rotation_matrix) + np.array([2.3354, 112.8349, -193.7355])
    model_pc = np.loadtxt(model_point_cloud_path)

    print(f"实际加工模型点云数量：{len(source_pc)}")
    print(f"理论模型点云数量：{len(model_pc)}")

    # PCA粗配准
    aligned_source, rotation, translation = align_point_cloud(source_pc, model_pc)
    print("旋转矩阵:\n", rotation)
    print("平移向量:\n", translation)
    # 可视化
    vtk_visualize(source_pc, model_pc)
    vtk_visualize(aligned_source, model_pc)
