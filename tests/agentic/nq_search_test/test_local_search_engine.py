"""测试本地搜索引擎连接和功能

这个脚本用于测试本地搜索引擎是否能正常工作
"""

import sys
import json
import requests
from pathlib import Path
from typing import Dict, List, Any

# 将项目根目录添加到 Python 路径
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))


class LocalSearchEngineTester:
    """本地搜索引擎测试器"""

    def __init__(self,
                 service_url: str = "http://172.26.104.240:30002",
                 timeout: int = 30):
        """
        初始化测试器

        Args:
            service_url: 搜索服务URL
            timeout: 请求超时时间（秒）
        """
        self.service_url = service_url
        self.timeout = timeout
        print(f"初始化搜索引擎测试器")
        print(f"服务URL: {service_url}")
        print(f"超时时间: {timeout}秒")
        print("-" * 50)

    def test_connection(self) -> bool:
        """测试服务连接"""
        print("\n[测试1] 检查服务连接...")
        try:
            # 尝试连接服务的健康检查端点
            response = requests.get(
                f"{self.service_url}/health",
                timeout=5
            )
            if response.status_code == 200:
                print(f"✅ 服务连接成功! 状态码: {response.status_code}")
                return True
            else:
                print(f"⚠️ 服务响应异常，状态码: {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            print(f"❌ 无法连接到服务: {self.service_url}")
            print("   请检查：")
            print("   1. 服务是否已启动")
            print("   2. URL是否正确")
            print("   3. 网络是否可达")
            return False
        except requests.exceptions.Timeout:
            print(f"❌ 连接超时")
            return False
        except Exception as e:
            print(f"❌ 连接失败: {str(e)}")
            return False

    def test_search_api(self, query: str = "What is machine learning?") -> Dict[str, Any]:
        """
        测试搜索API

        Args:
            query: 测试查询

        Returns:
            搜索结果
        """
        print(f"\n[测试2] 测试搜索功能...")
        print(f"查询: {query}")

        try:
            # 构造搜索请求
            request_data = {
                "tool_name": "search",
                "arguments": {
                    "query": query
                }
            }

            # 发送请求
            response = requests.post(
                f"{self.service_url}/execute",
                json=request_data,
                timeout=self.timeout
            )

            if response.status_code == 200:
                result = response.json()
                print(f"✅ 搜索成功!")

                # 解析结果
                if "result" in result:
                    search_results = result["result"]
                    if isinstance(search_results, str):
                        # 尝试解析为JSON
                        try:
                            search_results = json.loads(search_results)
                        except:
                            pass

                    # 显示结果摘要
                    if isinstance(search_results, list):
                        print(f"   返回了 {len(search_results)} 条结果")
                        for i, item in enumerate(search_results[:3], 1):
                            if isinstance(item, dict):
                                title = item.get("title", "无标题")
                                print(f"   {i}. {title[:50]}...")
                    else:
                        print(f"   结果类型: {type(search_results)}")
                        print(f"   结果预览: {str(search_results)[:100]}...")

                return result
            else:
                print(f"❌ 搜索失败，状态码: {response.status_code}")
                print(f"   响应: {response.text[:200]}")
                return {}

        except requests.exceptions.ConnectionError:
            print(f"❌ 无法连接到搜索服务")
            return {}
        except requests.exceptions.Timeout:
            print(f"❌ 搜索请求超时（超过{self.timeout}秒）")
            return {}
        except Exception as e:
            print(f"❌ 搜索失败: {str(e)}")
            return {}

    def test_with_mock_api(self) -> bool:
        """测试使用模拟API的情况"""
        print("\n[测试3] 测试模拟API模式...")

        from roll.agentic.env.search import SearchEnv, SearchEnvConfig

        try:
            # 创建使用模拟API的环境
            config = SearchEnvConfig(
                dataset_path=None,  # 不加载数据集
                use_mock_api=True,  # 使用模拟API
                use_remote_service=False,  # 不使用远程服务
                disable_limiter=True,
                max_steps=5,
                max_search_calls=3
            )

            env = SearchEnv(config)

            # 执行模拟搜索
            mock_results = env.search("test query")

            print(f"✅ 模拟API工作正常")
            print(f"   返回了 {len(mock_results)} 条模拟结果")

            env.close()
            return True

        except Exception as e:
            print(f"❌ 模拟API测试失败: {str(e)}")
            return False

    def test_with_remote_service(self) -> bool:
        """测试使用远程服务的情况"""
        print("\n[测试4] 测试远程服务模式...")

        from roll.agentic.env.search import SearchEnv, SearchEnvConfig

        try:
            # 创建使用远程服务的环境
            config = SearchEnvConfig(
                dataset_path=None,  # 不加载数据集
                use_mock_api=False,  # 不使用模拟API
                use_remote_service=True,  # 使用远程服务
                remote_service_url=self.service_url,
                remote_service_timeout=self.timeout,
                disable_limiter=True,
                max_steps=5,
                max_search_calls=3
            )

            env = SearchEnv(config)

            # 执行真实搜索
            real_results = env.search("What is Python programming?")

            print(f"✅ 远程服务工作正常")
            print(f"   返回了 {len(real_results)} 条真实结果")

            env.close()
            return True

        except Exception as e:
            print(f"❌ 远程服务测试失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def run_all_tests(self):
        """运行所有测试"""
        print("=" * 50)
        print("开始测试本地搜索引擎")
        print("=" * 50)

        results = {}

        # 测试1：连接测试
        results['connection'] = self.test_connection()

        # 测试2：搜索API测试
        if results['connection']:
            search_result = self.test_search_api()
            results['search_api'] = bool(search_result)
        else:
            print("\n[测试2] 跳过搜索API测试（连接失败）")
            results['search_api'] = False

        # 测试3：模拟API测试
        results['mock_api'] = self.test_with_mock_api()

        # 测试4：远程服务测试
        if results['connection']:
            results['remote_service'] = self.test_with_remote_service()
        else:
            print("\n[测试4] 跳过远程服务测试（连接失败）")
            results['remote_service'] = False

        # 总结
        print("\n" + "=" * 50)
        print("测试结果总结")
        print("=" * 50)
        for test_name, passed in results.items():
            status = "✅ 通过" if passed else "❌ 失败"
            print(f"{test_name:20} : {status}")

        # 给出建议
        print("\n" + "=" * 50)
        print("建议")
        print("=" * 50)

        if not results['connection']:
            print("⚠️ 无法连接到搜索服务，建议：")
            print("   1. 确认服务器 172.26.104.240:30002 是否可达")
            print("   2. 检查防火墙设置")
            print("   3. 确认搜索服务是否已启动")
            print("   4. 可以先使用 use_mock_api=True 进行开发测试")
        elif not results['search_api']:
            print("⚠️ 搜索API不工作，建议：")
            print("   1. 检查API端点是否正确")
            print("   2. 验证请求格式是否符合要求")
            print("   3. 查看服务器日志了解详细错误")
        else:
            print("✅ 搜索引擎工作正常，可以进行NQ Search任务训练！")

        return all(results.values())


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='测试本地搜索引擎')
    parser.add_argument('--url', type=str,
                       default='http://172.26.104.240:30002',
                       help='搜索服务URL')
    parser.add_argument('--timeout', type=int, default=30,
                       help='请求超时时间（秒）')
    parser.add_argument('--query', type=str,
                       default='What is machine learning?',
                       help='测试查询')

    args = parser.parse_args()

    # 创建测试器
    tester = LocalSearchEngineTester(
        service_url=args.url,
        timeout=args.timeout
    )

    # 运行所有测试
    success = tester.run_all_tests()

    # 返回状态码
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()