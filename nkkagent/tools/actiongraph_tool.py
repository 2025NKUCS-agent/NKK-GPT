#!/usr/bin/env python

import asyncio
import json
import os
import datetime
import aiofiles  # 添加aiofiles导入
from pathlib import Path
from typing import Dict, List, Optional, TypedDict, Union, Any, Type

from nkkagent.tools.base import BaseTool

# 定义内存文件路径，使用环境变量或默认路径
script_dir = Path(__file__).parent
default_memory_path = script_dir / 'actionnode.json'

# 如果ACTION_NODE_FILE_PATH只是文件名，则放在与脚本相同的目录中
ACTION_NODE_FILE_PATH = os.environ.get('ACTION_NODE_FILE_PATH')
if ACTION_NODE_FILE_PATH:
    memory_path = Path(ACTION_NODE_FILE_PATH)
    if not memory_path.is_absolute():
        memory_path = script_dir / ACTION_NODE_FILE_PATH
else:
    memory_path = default_memory_path

# 定义数据结构
class ActionNode(TypedDict):
    # Action Context
    context: str  # all the context, including all necessary info
    name: str
    entityType: str
    observations: List[str]
    # Action Output
    content: str
    children: Dict[str, str]  # 子节点的引用，键为关系名称，值为节点名称
    params: Dict[str, str]  # 输入参数的字典，键为参数名，值为参数类型的字符串表示

class NodeRelation(TypedDict):
    from_: str  # 使用from_避免与Python关键字冲突
    to: str
    relationType: str

class ActionNodeGraph(TypedDict):
    nodes: List[ActionNode]
    relations: List[NodeRelation]


class ActionNodeGraphTool(BaseTool):
    name: str = "action_node_graph"
    description: str = "管理动作节点图的工具，支持创建、查询、更新和删除动作节点和关系，用于存储和执行动作流程"
    parameters: dict = {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "description": "要执行的操作类型",
                "enum": [
                    "create_nodes",
                    "create_relations",
                    "add_observations",
                    "update_node_content",
                    "update_node_params",
                    "add_children",
                    "delete_nodes",
                    "delete_observations",
                    "delete_relations",
                    "read_graph",
                    "search_nodes",
                    "open_nodes",
                    "execute_node"
                ]
            },
            "nodes": {
                "type": "array",
                "description": "用于创建动作节点的数据列表",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "entityType": {"type": "string"},
                        "context": {"type": "string"},
                        "observations": {"type": "array", "items": {"type": "string"}},
                        "content": {"type": "string"},
                        "children": {"type": "object"},
                        "params": {"type": "object"}
                    },
                    "required": ["name", "entityType", "context"]
                }
            },
            "relations": {
                "type": "array",
                "description": "用于创建或删除关系的数据列表",
                "items": {
                    "type": "object",
                    "properties": {
                        "from": {"type": "string"},
                        "to": {"type": "string"},
                        "relationType": {"type": "string"}
                    },
                    "required": ["from", "to", "relationType"]
                }
            },
            "observations": {
                "type": "array",
                "description": "用于添加观察的数据列表",
                "items": {
                    "type": "object",
                    "properties": {
                        "nodeName": {"type": "string"},
                        "contents": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["nodeName", "contents"]
                }
            },
            "nodeNames": {
                "type": "array",
                "description": "要删除的节点名称列表",
                "items": {"type": "string"}
            },
            "contentUpdates": {
                "type": "array",
                "description": "用于更新节点内容的数据列表",
                "items": {
                    "type": "object",
                    "properties": {
                        "nodeName": {"type": "string"},
                        "content": {"type": "string"}
                    },
                    "required": ["nodeName", "content"]
                }
            },
            "paramUpdates": {
                "type": "array",
                "description": "用于更新节点参数的数据列表",
                "items": {
                    "type": "object",
                    "properties": {
                        "nodeName": {"type": "string"},
                        "params": {"type": "object"}
                    },
                    "required": ["nodeName", "params"]
                }
            },
            "childrenUpdates": {
                "type": "array",
                "description": "用于添加子节点的数据列表",
                "items": {
                    "type": "object",
                    "properties": {
                        "nodeName": {"type": "string"},
                        "children": {"type": "object"}
                    },
                    "required": ["nodeName", "children"]
                }
            },
            "deletions": {
                "type": "array",
                "description": "用于删除观察的数据列表",
                "items": {
                    "type": "object",
                    "properties": {
                        "nodeName": {"type": "string"},
                        "observations": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["nodeName", "observations"]
                }
            },
            "query": {
                "type": "string",
                "description": "用于搜索节点的查询字符串"
            },
            "names": {
                "type": "array",
                "description": "要打开的节点名称列表",
                "items": {"type": "string"}
            },
            "executeNodeName": {
                "type": "string",
                "description": "要执行的节点名称"
            },
            "executeParams": {
                "type": "object",
                "description": "执行节点时传递的参数"
            }
        },
        "required": ["operation"]
    }

    async def execute(self, **kwargs) -> Any:
        """执行动作节点图操作"""
        operation = kwargs.get("operation")

        if operation == "create_nodes":
            return await self._create_nodes(kwargs.get("nodes", []))
        elif operation == "create_relations":
            return await self._create_relations(kwargs.get("relations", []))
        elif operation == "add_observations":
            return await self._add_observations(kwargs.get("observations", []))
        elif operation == "update_node_content":
            return await self._update_node_content(kwargs.get("contentUpdates", []))
        elif operation == "update_node_params":
            return await self._update_node_params(kwargs.get("paramUpdates", []))
        elif operation == "add_children":
            return await self._add_children(kwargs.get("childrenUpdates", []))
        elif operation == "delete_nodes":
            return await self._delete_nodes(kwargs.get("nodeNames", []))
        elif operation == "delete_observations":
            return await self._delete_observations(kwargs.get("deletions", []))
        elif operation == "delete_relations":
            return await self._delete_relations(kwargs.get("relations", []))
        elif operation == "read_graph":
            return await self._read_graph()
        elif operation == "search_nodes":
            return await self._search_nodes(kwargs.get("query", ""))
        elif operation == "open_nodes":
            return await self._open_nodes(kwargs.get("names", []))
        elif operation == "execute_node":
            return await self._execute_node(
                kwargs.get("executeNodeName", ""),
                kwargs.get("executeParams", {})
            )
        else:
            raise ValueError(f"未知操作: {operation}")

    async def _load_graph(self) -> ActionNodeGraph:
        """加载动作节点图"""
        try:
            async with aiofiles.open(memory_path, "r") as file:
                data = await file.read()
                lines = [line for line in data.split("\n") if line.strip()]
                graph: ActionNodeGraph = {"nodes": [], "relations": []}

                for line in lines:
                    item = json.loads(line)
                    if item.get("type") == "node":
                        # 移除type字段并添加到nodes
                        node_data = {k: v for k, v in item.items() if k != "type"}
                        graph["nodes"].append(node_data)
                    if item.get("type") == "relation":
                        # 处理from字段，在Python中是关键字
                        relation_data = {}
                        for k, v in item.items():
                            if k != "type":
                                if k == "from":
                                    relation_data["from_"] = v
                                else:
                                    relation_data[k] = v
                        graph["relations"].append(relation_data)
                return graph
        except FileNotFoundError:
            return {"nodes": [], "relations": []}
        except Exception as e:
            raise e

    async def _save_graph(self, graph: ActionNodeGraph) -> None:
        """保存动作节点图"""
        lines = []
        # 处理节点
        for node in graph["nodes"]:
            node_copy = node.copy()
            node_json = {"type": "node", **node_copy}
            lines.append(json.dumps(node_json))

        # 处理关系，注意from_字段需要转换回from
        for relation in graph["relations"]:
            relation_copy = relation.copy()
            # 将from_转换回from
            if "from_" in relation_copy:
                relation_copy["from"] = relation_copy.pop("from_")
            relation_json = {"type": "relation", **relation_copy}
            lines.append(json.dumps(relation_json))

        async with aiofiles.open(memory_path, "w") as file:
            await file.write("\n".join(lines))

    async def _create_nodes(self, nodes: List[Dict]) -> List[ActionNode]:
        """创建动作节点"""
        graph = await self._load_graph()
        # 设置默认值
        for node in nodes:
            if "observations" not in node:
                node["observations"] = []
            if "content" not in node:
                node["content"] = ""
            if "children" not in node:
                node["children"] = {}
            if "params" not in node:
                node["params"] = {}

        new_nodes = [n for n in nodes if not any(existing["name"] == n["name"] for existing in graph["nodes"])]
        graph["nodes"].extend(new_nodes)
        await self._save_graph(graph)
        return new_nodes

    async def _create_relations(self, relations: List[Dict]) -> List[NodeRelation]:
        """创建关系"""
        graph = await self._load_graph()

        # 将from转换为from_
        processed_relations = []
        for relation in relations:
            relation_copy = relation.copy()
            if "from" in relation_copy:
                relation_copy["from_"] = relation_copy.pop("from")
            processed_relations.append(relation_copy)

        # 将from_转换为from进行比较
        def relation_exists(r: NodeRelation) -> bool:
            r_from = r.get("from_", "")
            for existing in graph["relations"]:
                existing_from = existing.get("from_", "")
                if (existing_from == r_from and
                    existing["to"] == r["to"] and
                    existing["relationType"] == r["relationType"]):
                    return True
            return False

        new_relations = [r for r in processed_relations if not relation_exists(r)]
        graph["relations"].extend(new_relations)
        await self._save_graph(graph)
        return new_relations

    async def _add_observations(self, observations: List[Dict[str, Union[str, List[str]]]]) -> List[Dict[str, Union[str, List[str]]]]:
        """添加观察"""
        graph = await self._load_graph()
        results = []

        for obs in observations:
            node_name = obs["nodeName"]
            contents = obs["contents"]

            # 查找节点
            node = next((n for n in graph["nodes"] if n["name"] == node_name), None)
            if not node:
                raise ValueError(f"Node with name {node_name} not found")

            # 添加新的观察
            new_observations = [content for content in contents if content not in node["observations"]]
            node["observations"].extend(new_observations)

            results.append({"nodeName": node_name, "addedObservations": new_observations})

        await self._save_graph(graph)
        return results

    async def _update_node_content(self, content_updates: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """更新节点内容"""
        graph = await self._load_graph()
        results = []

        for update in content_updates:
            node_name = update["nodeName"]
            content = update["content"]

            # 查找节点
            node = next((n for n in graph["nodes"] if n["name"] == node_name), None)
            if not node:
                raise ValueError(f"Node with name {node_name} not found")

            # 更新内容
            old_content = node.get("content", "")
            node["content"] = content

            results.append({
                "nodeName": node_name,
                "oldContent": old_content,
                "newContent": content
            })

        await self._save_graph(graph)
        return results

    async def _update_node_params(self, param_updates: List[Dict[str, Union[str, Dict]]]) -> List[Dict[str, Union[str, Dict]]]:
        """更新节点参数"""
        graph = await self._load_graph()
        results = []

        for update in param_updates:
            node_name = update["nodeName"]
            params = update["params"]

            # 查找节点
            node = next((n for n in graph["nodes"] if n["name"] == node_name), None)
            if not node:
                raise ValueError(f"Node with name {node_name} not found")

            # 更新参数
            old_params = node.get("params", {})
            node["params"] = {**old_params, **params}  # 合并参数

            results.append({
                "nodeName": node_name,
                "oldParams": old_params,
                "newParams": node["params"]
            })

        await self._save_graph(graph)
        return results

    async def _add_children(self, children_updates: List[Dict[str, Union[str, Dict]]]) -> List[Dict[str, Union[str, Dict]]]:
        """添加子节点"""
        graph = await self._load_graph()
        results = []

        for update in children_updates:
            node_name = update["nodeName"]
            children = update["children"]

            # 查找节点
            node = next((n for n in graph["nodes"] if n["name"] == node_name), None)
            if not node:
                raise ValueError(f"Node with name {node_name} not found")

            # 更新子节点
            old_children = node.get("children", {})
            node["children"] = {**old_children, **children}  # 合并子节点

            # 为每个新的子节点创建关系
            for rel_type, child_name in children.items():
                # 检查目标节点是否存在
                target_exists = any(n["name"] == child_name for n in graph["nodes"])
                if not target_exists:
                    continue  # 如果目标节点不存在，跳过创建关系

                # 创建新关系
                new_relation = {
                    "from_": node_name,
                    "to": child_name,
                    "relationType": rel_type
                }

                # 检查关系是否已存在
                if not any(
                    (r.get("from_") == node_name and
                     r.get("to") == child_name and
                     r.get("relationType") == rel_type)
                    for r in graph["relations"]
                ):
                    graph["relations"].append(new_relation)

            results.append({
                "nodeName": node_name,
                "oldChildren": old_children,
                "newChildren": node["children"]
            })

        await self._save_graph(graph)
        return results

    async def _delete_nodes(self, node_names: List[str]) -> Dict[str, str]:
        """删除节点"""
        graph = await self._load_graph()

        # 过滤节点
        graph["nodes"] = [n for n in graph["nodes"] if n["name"] not in node_names]

        # 过滤关系
        graph["relations"] = [r for r in graph["relations"]
                            if r.get("from_") not in node_names and r["to"] not in node_names]

        await self._save_graph(graph)
        return {"status": "success", "message": "节点已成功删除"}

    async def _delete_observations(self, deletions: List[Dict[str, Union[str, List[str]]]]) -> Dict[str, str]:
        """删除观察"""
        graph = await self._load_graph()

        for deletion in deletions:
            node_name = deletion["nodeName"]
            observations_to_delete = deletion["observations"]

            # 查找并更新节点
            node = next((n for n in graph["nodes"] if n["name"] == node_name), None)
            if node:
                node["observations"] = [o for o in node["observations"] if o not in observations_to_delete]

        await self._save_graph(graph)
        return {"status": "success", "message": "观察已成功删除"}

    async def _delete_relations(self, relations: List[Dict]) -> Dict[str, str]:
        """删除关系"""
        graph = await self._load_graph()

        # 将from转换为from_
        processed_relations = []
        for relation in relations:
            relation_copy = relation.copy()
            if "from" in relation_copy:
                relation_copy["from_"] = relation_copy.pop("from")
            processed_relations.append(relation_copy)

        # 过滤关系
        def should_delete(r: NodeRelation) -> bool:
            for del_relation in processed_relations:
                del_from = del_relation.get("from_", "")
                r_from = r.get("from_", "")
                if (r_from == del_from and
                    r["to"] == del_relation["to"] and
                    r["relationType"] == del_relation["relationType"]):
                    return True
            return False

        # 过滤关系
        old_relations = graph["relations"].copy()
        graph["relations"] = [r for r in graph["relations"] if not should_delete(r)]

        # 同时更新相关节点的children
        deleted_relations = [r for r in old_relations if should_delete(r)]
        for relation in deleted_relations:
            from_node = next((n for n in graph["nodes"] if n["name"] == relation.get("from_")), None)
            if from_node and relation["relationType"] in from_node.get("children", {}):
                if from_node["children"][relation["relationType"]] == relation["to"]:
                    del from_node["children"][relation["relationType"]]

        await self._save_graph(graph)
        return {"status": "success", "message": "关系已成功删除"}

    async def _read_graph(self) -> ActionNodeGraph:
        """读取整个动作节点图"""
        return await self._load_graph()

    async def _search_nodes(self, query: str) -> ActionNodeGraph:
        """搜索节点"""
        graph = await self._load_graph()
        query = query.lower()

        # 过滤节点
        filtered_nodes = [n for n in graph["nodes"] if
                         query in n["name"].lower() or
                         query in n["entityType"].lower() or
                         query in n["context"].lower() or
                         any(query in obs.lower() for obs in n["observations"]) or
                         query in n.get("content", "").lower()]

        # 创建过滤后的节点名称集合，用于快速查找
        filtered_node_names = {n["name"] for n in filtered_nodes}

        # 过滤关系，只包含过滤后节点之间的关系
        filtered_relations = [r for r in graph["relations"] if
                             r.get("from_") in filtered_node_names and
                             r["to"] in filtered_node_names]

        return {"nodes": filtered_nodes, "relations": filtered_relations}

    async def _open_nodes(self, names: List[str]) -> ActionNodeGraph:
        """打开指定节点"""
        graph = await self._load_graph()

        # 过滤节点
        filtered_nodes = [n for n in graph["nodes"] if n["name"] in names]

        # 创建过滤后的节点名称集合，用于快速查找
        filtered_node_names = {n["name"] for n in filtered_nodes}

        # 过滤关系，只包含过滤后节点之间的关系
        filtered_relations = [r for r in graph["relations"] if
                             r.get("from_") in filtered_node_names and
                             r["to"] in filtered_node_names]

        return {"nodes": filtered_nodes, "relations": filtered_relations}

    async def _execute_node(self, node_name: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """执行节点，可以通过LLM处理节点内容并执行"""
        graph = await self._load_graph()

        # 查找节点
        node = next((n for n in graph["nodes"] if n["name"] == node_name), None)
        if not node:
            raise ValueError(f"Node with name {node_name} not found")

        params = params or {}

        # 这里可以实现节点执行的逻辑
        # 例如：解析节点内容，调用LLM等
        result = {
            "nodeName": node_name,
            "executed": True,
            "params": params,
            "result": f"执行节点 {node_name} 的内容: {node.get('content', '(无内容)')}"
        }

        # 记录执行结果到观察中
        execution_obs = f"执行时间: {datetime.datetime.now().isoformat()} - 参数: {json.dumps(params)}"
        node["observations"].append(execution_obs)

        await self._save_graph(graph)
        return result
