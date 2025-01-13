import numpy as np
import vtk
from vtkmodules.util.numpy_support import numpy_to_vtk
from scipy.spatial import KDTree

# 渲染配置
render_point_cloud = True # 渲染点云
render_stl = False # 渲染STL模型

def load_actors(point_data, stl_data):
    actors = []
    
    if render_point_cloud:
        point_actor = load_point_cloud(point_data)
        actors.append(point_actor)
    
    if render_stl:
        stl_actor = load_stl_model(stl_data)
        actors.append(stl_actor)

    if not actors:
        print("错误: 没有任何数据被加载和渲染。请检查配置变量和文件路径。")
        exit(1)
    
    return actors

def load_point_cloud(source_data):

    # 新建 vtkPoints 实例
    points = vtk.vtkPoints()
    # 将 numpy 数据转换为 vtk 数据并设置到 vtkPoints
    vtk_data = numpy_to_vtk(source_data, deep=True)
    vtk_data.SetNumberOfComponents(3)  # 确保每个点有三个坐标分量
    points.SetData(vtk_data)

    # 新建 vtkPolyData 实例并设置点坐标
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)

    # 创建一个基于高度的颜色映射
    colors = vtk.vtkUnsignedCharArray()
    colors.SetNumberOfComponents(3)
    colors.SetName("Colors")

    z_values = source_data[:, 2]
    z_min, z_max = z_values.min(), z_values.max()

    for z in z_values:
        normalized_z = (z - z_min) / (z_max - z_min) if z_max > z_min else 0
        r = int(normalized_z * 255)
        g = 0
        b = int((1-normalized_z) * 255)
        colors.InsertNextTuple3(r, g, b)

    polydata.GetPointData().SetScalars(colors)

    # 使用 vtkVertexGlyphFilter 将点转化为可渲染的顶点
    vertex_filter = vtk.vtkVertexGlyphFilter()
    vertex_filter.SetInputData(polydata)
    vertex_filter.Update()

    # 创建点云的 mapper
    point_mapper = vtk.vtkPolyDataMapper()
    point_mapper.SetInputConnection(vertex_filter.GetOutputPort())

    # 创建点云的 actor
    point_actor = vtk.vtkActor()
    point_actor.SetMapper(point_mapper)
    # point_actor.GetProperty().SetColor(1, 0, 1)  # 绿色点
    point_actor.GetProperty().SetPointSize(4)     # 设置点大小

    return point_actor    

def load_stl_model(file_path):
    # 创建 STL 读取器
    stl_reader = vtk.vtkSTLReader()
    stl_reader.SetFileName(file_path)
    stl_reader.Update()
    
    # 创建 STL 的 mapper
    stl_mapper = vtk.vtkPolyDataMapper()
    stl_mapper.SetInputConnection(stl_reader.GetOutputPort())
    # 创建 STL 的 actor
    stl_actor = vtk.vtkActor()
    stl_actor.SetMapper(stl_mapper)
    stl_actor.GetProperty().SetColor(1, 0, 0)  # 红色
    stl_actor.GetProperty().SetOpacity(0.5)    # 可选：设置透明度

    return stl_actor

def visualization(actors):
    # 创建渲染器
    renderer = vtk.vtkRenderer()

    # 创建渲染窗口
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(1200, 1200)
    render_window.SetWindowName('点云与 STL 可视化')

    # 创建渲染窗口交互器
    render_interactor = vtk.vtkRenderWindowInteractor()
    render_interactor.SetRenderWindow(render_window)

    # 设置交互样式（可选）
    render_interactor.SetInteractorStyle(vtk.vtkInteractorStyleMultiTouchCamera())

    # 添加要渲染的元素
    for actor in actors:
        renderer.AddActor(actor)

    # 加一个坐标轴
    axes = vtk.vtkAxesActor()
    axes.SetTotalLength(40, 40, 40)
    axes.AxisLabelsOn()
    renderer.AddActor(axes)

    renderer.SetBackground(122, 122, 122)  # 黑色背景

    # 渲染循环
    render_interactor.Initialize()
    render_window.Render()
    render_interactor.Start()

def denoise_point_cloud(points):

    # 找到目标点最近的182个点
    kdtree = KDTree(points)
    query_point = [4.76, -71.17, 424.26]
    points_num = 182
    distances, indeces = kdtree.query(query_point, k=points_num)

    denoised_point_cloud = np.delete(points, indeces, axis=0)

    return denoised_point_cloud

def bilateral_filter_point_cloud(points, sigma_s=1.0, sigma_r=0.1, k=16):
    """
    使用双边滤波对点云进行降噪。
    
    参数:
    - points: 输入点云，形状为 (N, 3)。
    - sigma_spatial: 空间权重的标准差。
    - sigma_range: 属性（高度）权重的标准差。
    - k_neighbors: 每个点的邻居数量。
    
    返回:
    - denoised_points: 滤波后的点云，形状为 (N, 3)。
    """
    tree = KDTree(points)
    denoised_points = np.zeros_like(points)
    N = points.shape[0]
    
    for i in range(N):
        # 查找k个最近邻
        distances, indices = tree.query(points[i], k=k)
        neighbors = points[indices]
        
        # 计算空间权重
        w_spatial = np.exp(-(distances ** 2) / (2 * sigma_s ** 2))
        
        # 计算属性权重（这里使用Z坐标）
        z_diff = neighbors[:, 2] - points[i, 2]
        w_range = np.exp(-(z_diff ** 2) / (2 * sigma_r ** 2))
        
        # 总权重
        weights = w_spatial * w_range
        
        # 归一化权重
        weights /= np.sum(weights)
        
        # 计算新的点位置
        denoised_points[i] = np.sum(neighbors * weights[:, np.newaxis], axis=0)
        
        if i % 10000 == 0 and i > 0:
            print(f"处理进度: {i}/{N} 点")

    return denoised_points

if __name__ == '__main__':

    # 文件路径
    point_cloud_path = "src\\CalibrationPointCloud.txt"
    stl_path = "src\\YH0700000__1756220035.stl"

    point_cloud = np.loadtxt(point_cloud_path)
    stl_data = stl_path

    print(f"初始点云数量：{len(point_cloud)}")

    # 点云滤波
    denoised_point_cloud = denoise_point_cloud(point_cloud)
    # denoised_point_cloud = bilateral_filter_point_cloud(point_cloud, sigma_s=0.05, sigma_r=0.05, k=16)
    print(f"滤波后点云数量：{len(denoised_point_cloud)}")

    output_path = "src\\CalibrationPointCloud_denoised.txt"
    np.savetxt(output_path, denoised_point_cloud, fmt='%.6f')

    # 加载初始数据actor
    actors = []
    actors = load_actors(denoised_point_cloud, stl_data)

    # 渲染
    visualization(actors)

