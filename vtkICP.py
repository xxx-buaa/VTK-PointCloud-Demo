import numpy as np
import vtk
from vtkmodules.util.numpy_support import numpy_to_vtk
from scipy.spatial.transform import Rotation as R
from scipy.spatial import KDTree
from scipy.spatial import cKDTree

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

def align_icp(source, target, source_normals, rotation, translation, max_iterations=20, tolerance=1e-5, distance_threshold=0.01):
    """
    点到面ICP算法实现。
    
    参数:
    - source: 源点云 (N x 3)
    - target: 目标点云 (M x 3)
    - source_normals: 源点云法线 (N x 3)
    - max_iterations: 最大迭代次数
    - tolerance: 收敛阈值
    - distance-threshold: 距离阈值
    
    返回:
    - transformed_source: 变换后的源点云
    - final_rotation: 最终旋转矩阵
    - final_translation: 最终平移向量
    """
    # 初始化变换矩阵
    R_total = rotation
    t_total = translation
    transformed_source = source.copy()

    # 构建目标点云的KD树
    target_kdtree = cKDTree(target)

    prev_error = float('inf')

    for i in range(max_iterations):
        # 最近点匹配
        distances, indices = target_kdtree.query(transformed_source)
        valid_mask = distances < distance_threshold
        matched_target = target[indices[valid_mask]]
        matched_source = transformed_source[valid_mask]
        matched_normals = source_normals[valid_mask]

        if len(matched_target) == 0:
            print("匹配点不足，退出迭代")
            break

        # 计算误差向量 (点到平面)
        error_vectors = matched_target - matched_source
        normals_dot_error = np.sum(matched_normals * error_vectors, axis=1)

        # 构造优化方程 (Ax = b)
        A = np.hstack([
            np.cross(matched_source, matched_normals),
            matched_normals
        ])
        b = normals_dot_error

        # 求解最优变换
        x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        rotation_vector = x[:3]
        translation_vector = x[3:]

        # 构造旋转矩阵
        theta = np.linalg.norm(rotation_vector)
        if theta > 1e-8:
            axis = rotation_vector / theta
            skew = np.array([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]
            ])
            R_icp = np.eye(3) + np.sin(theta) * skew + (1 - np.cos(theta)) * (skew @ skew)
        else:
            R_icp = np.eye(3)

        t_icp = translation_vector

        # 更新总变换
        R_total = R_icp @ R_total
        t_total = R_icp @ t_total + t_icp

        # 应用变换
        transformed_source = (transformed_source @ R_icp.T) + t_icp

        # 计算平均误差
        mean_error = np.mean(np.abs(normals_dot_error))
        print(f"Iteration {i+1}: mean error = {mean_error}")
        if abs(prev_error - mean_error) < tolerance:
            print("ICP收敛")
            break
        prev_error = mean_error

    return transformed_source, R_total, t_total

def compute_normals(points, k=10):
    """
    使用批量处理和向量化优化的法线计算。
    参数:
        points: 点云数据 (N x 3)
        k: 每个点的邻域点数
    返回:
        normals: 法线数组 (N x 3)
    """
    # 构建 KD 树
    tree = cKDTree(points)

    # 批量查询所有点的 k 个最近邻索引
    _, indices = tree.query(points, k=k + 1)  # 包括自身，后续排除自身
    neighbors = points[indices[:, 1:]]  # 排除每个点自身的索引

    # 初始化法线数组
    normals = np.zeros_like(points)

    # 批量计算协方差矩阵和法线
    for i in range(len(points)):
        # 取邻域点
        neighbor_points = neighbors[i]

        # 计算协方差矩阵
        cov = np.cov(neighbor_points.T)

        # 求解最小特征值对应的特征向量
        _, eigvecs = np.linalg.eigh(cov)
        normal = eigvecs[:, 0]

        # 确保法线方向一致
        if np.dot(normal, points[i]) < 0:
            normal = -normal

        normals[i] = normal

    return normals

def voxel_downsample(points ,voxel_size):
    """
    使用体素网格对点云下采样。
    参数:
        points: 点云数据 (N x 3)
        voxel_size: 体素网格大小 (float)
    返回:
        downsampled_points: 下采样后的点云 (M x 3)
    """
    if voxel_size <= 0:
        raise ValueError("voxel_size must be greater than 0")

    # 将点云坐标量化为体素坐标
    voxel_indices = np.floor(points / voxel_size).astype(np.int32)

    # 创建体素索引到点的映射
    voxel_dict = {}
    for idx, voxel_idx in enumerate(voxel_indices):
        key = tuple(voxel_idx)
        if key not in voxel_dict:
            voxel_dict[key] = []
        voxel_dict[key].append(points[idx])

    # 计算每个体素的代表点（体素内点的平均值）
    downsampled_points = [np.mean(voxel_dict[key], axis=0) for key in voxel_dict]

    return np.array(downsampled_points)

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

    # 设置交互样式（可选）
    interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
    interactor.SetRenderWindow(render_window)

    # 启动可视化
    render_window.Render()
    interactor.Start()

if __name__ == '__main__':

    # 文件路径
    source_point_cloud_path = "src\\CalibrationPointCloud_denoised.txt"
    model_point_cloud_path = "src\\point_cloud.txt"

    source_pc = np.loadtxt(source_point_cloud_path)
    model_pc = np.loadtxt(model_point_cloud_path)

    
    print(f"实际加工模型点云数量：{len(source_pc)}")
    print(f"理论模型点云数量：{len(model_pc)}")

    # PCA粗配准
    aligned_source, rotation, translation = align_point_cloud(source_pc, model_pc)
    print("旋转矩阵:\n", rotation)
    print("平移向量:\n", translation)

    # 可视化
    print("显示粗配准结果...")
    vtk_visualize(aligned_source, model_pc)

    # 因为法线和点到面ICP算法计算量太大，加一个下采样环节
    print("开始下采样...")
    voxel_size = 0.3
    downsampled_source_pc = voxel_downsample(source_pc, voxel_size)
    downsampled_model_pc = voxel_downsample(model_pc, voxel_size)
    print(f"下采样后的点云点数: {len(downsampled_source_pc)}")
    print(f"下采样后的点云点数: {len(downsampled_model_pc)}")

    # 计算法线
    print("开始计算法线...")
    source_normal = compute_normals(downsampled_source_pc)

    # 精配准（ICP）
    print("开始ICP算法...")
    transformed_source, final_rotation, final_translation = align_icp(
        downsampled_source_pc, 
        downsampled_model_pc, 
        source_normal,
        rotation,
        translation,
        max_iterations=20,
        tolerance=1e-5,
        distance_threshold=0.01
    )
    print("ICP精配准结果:")
    print("最终旋转矩阵:\n", final_rotation)
    print("最终平移向量:\n", final_translation)

    # 可视化精配准结果
    print("显示ICP精配准结果...")
    vtk_visualize(transformed_source, model_pc)

    # 保存点云到TXT文件
    output_txt = "src\\CalibrationPointCloud_registrated.txt"
    # np.savetxt(output_txt, transformed_source, fmt='%.6f', delimiter='\t')
    print(f"点云已保存到 {output_txt}")