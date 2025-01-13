import vtk
import numpy as np

def read_stl(file_path):
    """读取STL文件并返回vtkPolyData对象"""
    reader = vtk.vtkSTLReader()
    reader.SetFileName(file_path)
    reader.Update()
    return reader.GetOutput()

def render_polydata(polydata):
    """渲染vtkPolyData对象"""
    # 创建Mapper
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    
    # 创建Actor
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    
    # 创建Renderer
    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(0.1, 0.2, 0.4)  # 设置背景颜色
    
    # 创建Render Window
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(800, 600)
    
    # 创建Render Window Interactor
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    
    # 开始渲染
    render_window.Render()
    interactor.Start()

def extract_points(polydata):
    """从vtkPolyData中提取点坐标并返回NumPy数组"""
    points_vtk = polydata.GetPoints()
    num_points = points_vtk.GetNumberOfPoints()
    points = np.array([points_vtk.GetPoint(i) for i in range(num_points)])
    return points

def sample_uniformly(polydata, num_points):
    """
    对vtkPolyData进行均匀采样，返回采样后的点的NumPy数组
    """
    # 获取所有三角形
    triangles = []
    for i in range(polydata.GetNumberOfCells()):
        cell = polydata.GetCell(i)
        if cell.GetCellType() == vtk.VTK_TRIANGLE:
            pt_ids = cell.GetPointIds()
            triangle = [polydata.GetPoint(pt_ids.GetId(j)) for j in range(3)]
            triangles.append(triangle)
    
    triangles = np.array(triangles)  # Shape: (num_triangles, 3, 3)
    
    # 计算每个三角形的面积
    def triangle_area(tri):
        a = tri[1] - tri[0]
        b = tri[2] - tri[0]
        cross = np.cross(a, b)
        return 0.5 * np.linalg.norm(cross, axis=1)
    
    # 计算所有三角形的面积
    areas = []
    for tri in triangles:
        a = np.subtract(tri[1], tri[0])
        b = np.subtract(tri[2], tri[0])
        cross = np.cross(a, b)
        area = 0.5 * np.linalg.norm(cross)
        areas.append(area)
    areas = np.array(areas)
    
    # 计算每个三角形的概率
    total_area = np.sum(areas)
    probabilities = areas / total_area
    
    # 选择要采样的三角形索引
    sampled_tri_indices = np.random.choice(len(triangles), size=num_points, p=probabilities)
    
    # 在选定的三角形上均匀采样点
    sampled_points = []
    for idx in sampled_tri_indices:
        tri = triangles[idx]
        # 生成两个随机数
        r1 = np.random.uniform(0, 1)
        r2 = np.random.uniform(0, 1)
        # 保证采样点均匀分布在三角形上
        if r1 + r2 > 1:
            r1 = 1 - r1
            r2 = 1 - r2
        point = tri[0] + r1 * (tri[1] - tri[0]) + r2 * (tri[2] - tri[0])
        sampled_points.append(point)
    
    sampled_points = np.array(sampled_points)
    return sampled_points

def save_point_cloud_to_txt(points, file_path):
    """将点云数据保存为TXT文件"""
    np.savetxt(file_path, points, fmt='%.6f', delimiter='\t')

def main():
    # 指定STL文件路径
    stl_file = "src\\CalibrationComparison.stl"  # 替换为你的STL文件路径
    
    # 读取STL文件
    polydata = read_stl(stl_file)
    
    # 渲染STL模型
    render_polydata(polydata)
    
    # 提取原始点数据
    original_points = extract_points(polydata)
    print(f"原始点数量: {original_points.shape[0]}")
    
    # 均匀化点云
    desired_num_points = 451205  # 设置所需点数
    uniform_points = sample_uniformly(polydata, desired_num_points)
    print(f"均匀化后的点数量: {uniform_points.shape[0]}")
    
    # 保存点云到TXT文件
    output_txt = "src\\point_cloud.txt"
    save_point_cloud_to_txt(uniform_points, output_txt)
    print(f"点云已保存到 {output_txt}")

if __name__ == "__main__":
    main()
