import sys
from pathlib import Path
from loguru import logger

def main():
    """主函数"""
    logger.info("开始测试配置系统...")
    
    # 添加项目根目录到 Python 路径
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
        logger.debug(f"添加项目根目录到 Python 路径: {project_root}")
    
    try:
        from config.config import config
        logger.info(f"项目根目录: {config.root_path}")
        logger.info(f"工作空间目录: {config.workspace_root}")
        logger.info(f"配置文件: {config._get_config_path()}")
    except Exception as e:
        logger.error(f"配置测试失败: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()
