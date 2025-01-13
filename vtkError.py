import numpy as np
from scipy.spatial import KDTree
import vtk
from vtkmodules.util.numpy_support import numpy_to_vtk
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkCommonCore import vtkFloatArray
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkPolyDataMapper,
    vtkRenderer,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
)
from vtkmodules.vtkFiltersSources import vtkSphereSource
from vtkmodules.vtkFiltersCore import vtkGlyph3D
    
if __name__ == '__main__':

    # ===========================
    # 1. 读取点云
    # ===========================
    # 文件路径
    source_point_cloud_path = "src\\CalibrationPointCloud_registrated.txt"
    model_point_cloud_path = "src\\point_cloud.txt"

    measured_pc = np.loadtxt(source_point_cloud_path)
    theoretical_pc = np.loadtxt(model_point_cloud_path)

    # 打印形状
    print(f"测量点云形状: {measured_pc.shape}")
    print(f"理论点云形状: {theoretical_pc.shape}")

    print(f"实际加工模型点云数量：{len(measured_pc)}")
    print(f"理论模型点云数量：{len(theoretical_pc)}")

    # ===========================
    # 2. 计算误差
    # ===========================
    # 理论模型的KDTree
    theoretical_tree = KDTree(theoretical_pc)

    # 对每个测量点找到最近的理论点（数量该怎么衡量？只能用1，不然会多开一个维度）
    distances, indices = theoretical_tree.query(measured_pc, k=1)

    # 计算平均误差
    average_error = np.mean(distances)
    print(f"平均误差：{average_error:.4f} mm")

    # ===========================
    # 3. 识别误差最大的5个区域
    # ===========================
    # 定义选择的误差点数
    num_top_points = 100  # 根据需要调整

    # 找出误差最大的前num_top_points个点的索引
    top_indices = np.argsort(distances)[-num_top_points:]
    top_distances = distances[top_indices]
    top_points = measured_pc[top_indices]

    # 打印形状
    print(f"top_points形状: {top_points.shape}")

    # ===========================
    # 4. 使用KDTree进行聚类
    # ===========================
    tree = KDTree(top_points)
    clusters = []
    visited = np.zeros(top_points.shape[0], dtype=bool)
    cluster_radius = 5.0  # 聚类半径，根据实际情况调整

    for i in range(top_points.shape[0]):
        if not visited[i]:
            # 查询在cluster_radius范围内的所有点
            indices = tree.query_ball_point(top_points[i], r=cluster_radius)
            clusters.append(indices)
            visited[indices] = True
            if len(clusters) >= 5:
                break

    # 如果找到的聚类少于5，可能需要增加num_top_points或调整cluster_radius
    if len(clusters) < 5:
        print("警告：找到的聚类少于5个，请调整参数。")

    # ===========================
    # 5. 计算每个聚类的中心点和误差
    # ===========================
    cluster_centers = []
    cluster_errors = []

    for cluster in clusters[:5]:
        cluster_points = top_points[cluster]
        center = np.mean(cluster_points, axis=0)
        cluster_error = np.mean(top_distances[cluster])
        cluster_centers.append(center)
        cluster_errors.append(cluster_error)

    print("误差最大的5个区域中心及其平均误差:")
    for i, (center, error) in enumerate(zip(cluster_centers, cluster_errors), 1):
        print(f"区域 {i}: 中心坐标 {center}, 平均误差 {error:.4f} mm")

    # ===========================
    # 6. 使用VTK可视化
    # ===========================

    # 创建颜色对象
    colors = vtkNamedColors()

    # ---------------------------
    # 6.1. 渲染点云
    # ---------------------------

    # 将点云数据转换为vtkPoints
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(measured_pc, deep=True, array_type=vtk.VTK_FLOAT))

    # 创建一个vtkPolyData对象并设置点
    point_polydata = vtk.vtkPolyData()
    point_polydata.SetPoints(points)

    # 使用 vtkGlyph3D 代替 vtkVertexGlyphFilter
    glyph_source = vtk.vtkSphereSource()
    glyph_source.SetRadius(0.1)  # 设置点的大小，根据点云尺度调整
    glyph_source.SetThetaResolution(8)
    glyph_source.SetPhiResolution(8)

    glyph = vtkGlyph3D()
    glyph.SetSourceConnection(glyph_source.GetOutputPort())
    glyph.SetInputData(point_polydata)
    glyph.ScalingOff()
    glyph.Update()

    # 创建映射器
    mapper = vtkPolyDataMapper()
    mapper.SetInputConnection(glyph.GetOutputPort())

    # 设置统一颜色（例如白色）
    mapper.ScalarVisibilityOff()  # 关闭标量可见性，以使用统一颜色

    # 创建点云的actor
    point_actor = vtkActor()
    point_actor.SetMapper(mapper)
    point_actor.GetProperty().SetColor(colors.GetColor3d("White"))
    point_actor.GetProperty().SetOpacity(1.0)  # 不透明
    point_actor.GetProperty().SetPointSize(2)

    # ---------------------------
    # 6.2. 渲染误差最大区域
    # ---------------------------

    # 创建一个列表来存储区域中心的actor
    region_actors = []

    for i, center in enumerate(cluster_centers):
        sphere = vtkSphereSource()
        sphere.SetCenter(center)
        sphere.SetRadius(2.0)  # 根据需要调整大小

        sphere_mapper = vtkPolyDataMapper()
        sphere_mapper.SetInputConnection(sphere.GetOutputPort())

        sphere_actor = vtkActor()
        sphere_actor.SetMapper(sphere_mapper)
        sphere_actor.GetProperty().SetColor(colors.GetColor3d("Yellow"))
        sphere_actor.GetProperty().SetOpacity(0.6)  # 半透明

        region_actors.append(sphere_actor)

    # ---------------------------
    # 6.3. 添加文本标签
    # ---------------------------

    # 创建文本标签的actor列表
    text_actors = []

    for i, center in enumerate(cluster_centers):
        text = vtk.vtkVectorText()
        text.SetText(f"区域 {i+1}")

        text_mapper = vtk.vtkPolyDataMapper()
        text_mapper.SetInputConnection(text.GetOutputPort())

        text_actor = vtkActor()
        text_actor.SetMapper(text_mapper)
        text_actor.SetScale(0.5, 0.5, 0.5)  # 根据需要调整大小
        text_actor.SetPosition(center + np.array([3, 3, 3]))  # 偏移位置避免与球体重叠
        text_actor.GetProperty().SetColor(colors.GetColor3d("Black"))

        text_actors.append(text_actor)

    # ---------------------------
    # 6.4. 创建渲染器、渲染窗口和交互器
    # ---------------------------

    renderer = vtkRenderer()
    renderer.GradientBackgroundOn()
    render_window = vtkRenderWindow()
    render_window.SetWindowName("Point Cloud Error Visualization")
    render_window.AddRenderer(renderer)
    interactor = vtkRenderWindowInteractor()

    # 设置交互样式（可选）
    interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
    interactor.SetRenderWindow(render_window)

    # ---------------------------
    # 6.5. 添加演员到渲染器
    # ---------------------------

    # 添加点云
    renderer.AddActor(point_actor)

    # 添加误差区域
    for sphere_actor in region_actors:
        renderer.AddActor(sphere_actor)

    # 添加文本标签
    for text_actor in text_actors:
        renderer.AddActor(text_actor)

    # 添加平均误差信息作为文本
    # 使用 vtkTextActor2D 以便更好地控制文本位置
    average_text_actor = vtk.vtkTextActor()
    average_text_actor.SetInput(f"平均误差: {average_error:.4f} mm")
    average_text_actor.GetTextProperty().SetFontSize(24)
    average_text_actor.GetTextProperty().SetColor(colors.GetColor3d("Black"))
    average_text_actor.SetDisplayPosition(10, 10)  # 屏幕坐标

    renderer.AddActor2D(average_text_actor)

    # ---------------------------
    # 6.6. 设置渲染器属性
    # ---------------------------

    # 设置背景颜色
    renderer.SetBackground(0.7, 0.85, 1.0)   # 起始颜色
    renderer.SetBackground2(0.5, 0.75, 1.0)  # 结束颜色

    # 设置相机视角
    renderer.ResetCamera()

    # ---------------------------
    # 6.7. 渲染并开始交互
    # ---------------------------

    render_window.Render()
    interactor.Start()