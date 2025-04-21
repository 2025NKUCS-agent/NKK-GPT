import os
import ast
import pkg_resources
from pathlib import Path
from typing import Set, List

def parse_imports(file_path: str) -> Set[str]:
    """解析单个Python文件中的所有导入语句"""
    imports = set()
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            tree = ast.parse(file.read())
            
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    imports.add(name.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module.split('.')[0])
    except Exception as e:
        print(f"无法解析文件 {file_path}: {str(e)}")
    
    return imports

def find_python_files(directory: str) -> List[str]:
    """递归查找所有Python文件"""
    python_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    return python_files

def get_installed_packages() -> Set[str]:
    """获取已安装的包及其版本"""
    return {pkg.key: pkg.version for pkg in pkg_resources.working_set}

def main():
    # 获取当前目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 查找所有Python文件
    python_files = find_python_files(current_dir)
    
    # 收集所有导入
    all_imports = set()
    for file in python_files:
        imports = parse_imports(file)
        all_imports.update(imports)
    
    # 获取已安装的包
    installed_packages = get_installed_packages()
    
    # 过滤掉标准库模块
    stdlib_modules = set(sys.stdlib_module_names)
    third_party_imports = {imp for imp in all_imports if imp not in stdlib_modules}
    
    # 生成requirements.txt
    with open('requirements.txt', 'w', encoding='utf-8') as f:
        for package in sorted(third_party_imports):
            if package in installed_packages:
                f.write(f"{package}=={installed_packages[package]}\n")
            else:
                f.write(f"{package}\n")
    
    print(f"已生成 requirements.txt，共找到 {len(third_party_imports)} 个第三方包。")

if __name__ == '__main__':
    import sys
    main() 