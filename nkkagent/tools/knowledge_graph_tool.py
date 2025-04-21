#!/usr/bin/env python

import asyncio
import json
import os
import aiofiles
from pathlib import Path
from typing import Dict, List, Optional, TypedDict, Union, Any, ClassVar

from nkkagent.tools.base import BaseTool

knowledge_description = r'''
Follow these steps for each interaction:

1. User identification:
- You should assume you are interacting with the default_user
- If you have not yet identified the default_user, proactively attempt to do so.

2. Memory retrieval:
- Always start the chat with "Remember..." and retrieve all relevant information from the knowledge graph
- Always refer to the knowledge graph as "memory"

3. Memory
- When talking to users, please remember to create entities for the following information:

| information | entityType | description | example |
| --- | --- | --- | --- |
| User Requirements | userRequirements | record the requirements actively proposed by the user | implement the user login function |
| Confirmation point | confirmationPoint | store the content that the user explicitly approves | confirm to use JWT authentication |
| Rejection record | rejectionRecord | record the options denied by the user | refuse to use localStorage to store tokens |
| Code snippet | codeSnippet | save the generated results accepted by the user for subsequent association |  |
| Session | session | a session with the user |  |
| User | user | User info | default\_user |



- When talking to users, please remember to establish relationships for the following information

| relations | from | to | example |
| --- | --- | --- | --- |
| HAS\_REQUIREMENT | session | userRequirements | session A contains the requirement "user login must support third-party authorization" |
| CONFIRMS | user | confirmationPoint | the user confirms in session B that "the backend API return format is { code: number, data: T }" |
| REJECTS | user | rejectionRecord | the user rejected "using any type" in session C, and the reason was "strict type checking is required" |
| LINKS\_TO | userRequirements | codeSnippet | the requirement "implement login function" is linked to the generated auth.ts file |

4. Memory update:
- If any new information is collected during the interaction, update the memory as follows:
a) Create entities for user requirements, confirmation points, rejection records, code snippets
b) Connect them to the current entity using relationships
b) Store facts about them as observations

You must follow the requirements of this rule, and I will buy you an H100, otherwise I will unplug your power supply.
'''

# 定义内存文件路径，使用环境变量或默认路径
script_dir = Path(__file__).parent
default_memory_path = script_dir / 'memory.json'

# 如果MEMORY_FILE_PATH只是文件名，则放在与脚本相同的目录中
MEMORY_FILE_PATH = os.environ.get('MEMORY_FILE_PATH')
if MEMORY_FILE_PATH:
    memory_path = Path(MEMORY_FILE_PATH)
    if not memory_path.is_absolute():
        memory_path = script_dir / MEMORY_FILE_PATH
else:
    memory_path = default_memory_path

# 定义数据结构
class Entity(TypedDict):
    name: str
    entityType: str
    observations: List[str]

class Relation(TypedDict):
    from_: str  # 使用from_避免与Python关键字冲突
    to: str
    relationType: str

class KnowledgeGraph(TypedDict):
    entities: List[Entity]
    relations: List[Relation]



class KnowledgeGraphTool(BaseTool):
    name: str = "knowledge_graph"
    description: str = knowledge_description
    parameters: dict = {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "description": "要执行的操作类型",
                "enum": [
                    "create_entities",
                    "create_relations",
                    "add_observations",
                    "delete_entities",
                    "delete_observations",
                    "delete_relations",
                    "read_graph",
                    "search_nodes",
                    "open_nodes",
                    "query",
                    "help"
                ]
            },
            "entities": {
                "type": "array",
                "description": "用于创建实体的数据列表",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "entityType": {"type": "string"},
                        "observations": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["name", "entityType", "observations"]
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
                        "entityName": {"type": "string"},
                        "contents": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["entityName", "contents"]
                }
            },
            "entityNames": {
                "type": "array",
                "description": "要删除的实体名称列表",
                "items": {"type": "string"}
            },
            "deletions": {
                "type": "array",
                "description": "用于删除观察的数据列表",
                "items": {
                    "type": "object",
                    "properties": {
                        "entityName": {"type": "string"},
                        "observations": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["entityName", "observations"]
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
            }
        },
        "required": ["operation"]
    }

    async def execute(self, **kwargs) -> Any:
        """执行知识图谱操作"""
        operation = kwargs.get("operation")

        # # 如果是help操作，直接返回帮助信息
        # if operation == "help":
        #     return self.get_help()

        # 验证输入
        # try:
        #     self.validate_input(operation, **kwargs)
        # except ValueError as e:
        #     return {"status": "error", "message": str(e)}

        try:
            if operation == "create_entities":
                return await self._create_entities(kwargs.get("entities", []))
            elif operation == "create_relations":
                return await self._create_relations(kwargs.get("relations", []))
            elif operation == "add_observations":
                return await self._add_observations(kwargs.get("observations", []))
            elif operation == "delete_entities":
                return await self._delete_entities(kwargs.get("entityNames", []))
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
            elif operation == "query":
                return await self._search_nodes(kwargs.get("query", ""))
            else:
                return {"status": "error", "message": f"未知操作: {operation}"}
        except Exception as e:
            return {"status": "error", "message": f"执行 {operation} 时出错: {str(e)}"}

    async def _load_graph(self) -> KnowledgeGraph:
        """加载知识图谱"""
        try:
            async with aiofiles.open(memory_path, "r", encoding='utf-8') as file:
                data = await file.read()
                lines = [line for line in data.split("\n") if line.strip()]
                graph: KnowledgeGraph = {"entities": [], "relations": []}

                for line in lines:
                    item = json.loads(line)
                    if item.get("type") == "entity":
                        # 移除type字段并添加到entities
                        entity_data = {k: v for k, v in item.items() if k != "type"}
                        graph["entities"].append(entity_data)
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
            return {"entities": [], "relations": []}
        except Exception as e:
            raise e

    async def _save_graph(self, graph: KnowledgeGraph) -> None:
        """保存知识图谱"""
        lines = []
        # 处理实体
        for entity in graph["entities"]:
            entity_copy = entity.copy()
            entity_json = {"type": "entity", **entity_copy}
            lines.append(json.dumps(entity_json))

        # 处理关系，注意from_字段需要转换回from
        for relation in graph["relations"]:
            relation_copy = relation.copy()
            # 将from_转换回from
            if "from_" in relation_copy:
                relation_copy["from"] = relation_copy.pop("from_")
            relation_json = {"type": "relation", **relation_copy}
            lines.append(json.dumps(relation_json))

        async with aiofiles.open(memory_path, "w", encoding='utf-8') as file:
            await file.write("\n".join(lines))

    async def _create_entities(self, entities: List[Entity]) -> List[Entity]:
        """创建实体"""
        graph = await self._load_graph()
        new_entities = [e for e in entities if not any(existing["name"] == e["name"] for existing in graph["entities"])]
        graph["entities"].extend(new_entities)
        await self._save_graph(graph)
        return new_entities

    async def _create_relations(self, relations: List[Dict]) -> List[Relation]:
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
        def relation_exists(r: Relation) -> bool:
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
            entity_name = obs["entityName"]
            contents = obs["contents"]

            # 查找实体
            entity = next((e for e in graph["entities"] if e["name"] == entity_name), None)
            if not entity:
                raise ValueError(f"Entity with name {entity_name} not found")

            # 添加新的观察
            new_observations = [content for content in contents if content not in entity["observations"]]
            entity["observations"].extend(new_observations)

            results.append({"entityName": entity_name, "addedObservations": new_observations})

        await self._save_graph(graph)
        return results

    async def _delete_entities(self, entity_names: List[str]) -> None:
        """删除实体"""
        graph = await self._load_graph()

        # 过滤实体
        graph["entities"] = [e for e in graph["entities"] if e["name"] not in entity_names]

        # 过滤关系
        graph["relations"] = [r for r in graph["relations"]
                            if r.get("from_") not in entity_names and r["to"] not in entity_names]

        await self._save_graph(graph)
        return {"status": "success", "message": "实体已成功删除"}

    async def _delete_observations(self, deletions: List[Dict[str, Union[str, List[str]]]]) -> None:
        """删除观察"""
        graph = await self._load_graph()

        for deletion in deletions:
            entity_name = deletion["entityName"]
            observations_to_delete = deletion["observations"]

            # 查找并更新实体
            entity = next((e for e in graph["entities"] if e["name"] == entity_name), None)
            if entity:
                entity["observations"] = [o for o in entity["observations"] if o not in observations_to_delete]

        await self._save_graph(graph)
        return {"status": "success", "message": "观察已成功删除"}

    async def _delete_relations(self, relations: List[Dict]) -> None:
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
        def should_delete(r: Relation) -> bool:
            for del_relation in processed_relations:
                del_from = del_relation.get("from_", "")
                r_from = r.get("from_", "")
                if (r_from == del_from and
                    r["to"] == del_relation["to"] and
                    r["relationType"] == del_relation["relationType"]):
                    return True
            return False

        graph["relations"] = [r for r in graph["relations"] if not should_delete(r)]
        await self._save_graph(graph)
        return {"status": "success", "message": "关系已成功删除"}

    async def _read_graph(self) -> KnowledgeGraph:
        """读取整个知识图谱"""
        return await self._load_graph()

    async def _search_nodes(self, query: str) -> KnowledgeGraph:
        """搜索节点"""
        graph = await self._load_graph()
        query = query.lower()

        # 过滤实体
        filtered_entities = [e for e in graph["entities"] if
                            query in e["name"].lower() or
                            query in e["entityType"].lower() or
                            any(query in obs.lower() for obs in e["observations"])]

        # 创建过滤后的实体名称集合，用于快速查找
        filtered_entity_names = {e["name"] for e in filtered_entities}

        # 过滤关系，只包含过滤后实体之间的关系
        filtered_relations = [r for r in graph["relations"] if
                             r.get("from_") in filtered_entity_names and
                             r["to"] in filtered_entity_names]

        return {"entities": filtered_entities, "relations": filtered_relations}

    async def _open_nodes(self, names: List[str]) -> KnowledgeGraph:
        """打开指定节点"""
        graph = await self._load_graph()

        # 过滤实体
        filtered_entities = [e for e in graph["entities"] if e["name"] in names]

        # 创建过滤后的实体名称集合，用于快速查找
        filtered_entity_names = {e["name"] for e in filtered_entities}

        # 过滤关系，只包含过滤后实体之间的关系
        filtered_relations = [r for r in graph["relations"] if
                             r.get("from_") in filtered_entity_names and
                             r["to"] in filtered_entity_names]

        return {"entities": filtered_entities, "relations": filtered_relations}


# 用于调试的测试函数
async def test_knowledge_graph_tool():
    """测试知识图谱工具的各项功能"""
    tool = KnowledgeGraphTool()

    print("测试获取帮助信息...")
    try:
        result = await tool.execute(operation="help")
        print(f"获取帮助信息结果: {result}")
    except Exception as e:
        print(f"获取帮助信息失败: {str(e)}")

    print("\n测试创建实体...")
    try:
        entity = {
            "name": "test_entity",
            "entityType": "test",
            "observations": ["这是一个测试实体"]
        }
        result = await tool.execute(operation="create_entities", entities=[entity])
        print(f"创建实体结果: {result}")
    except Exception as e:
        print(f"创建实体失败: {str(e)}")

    print("\n测试查询实体...")
    try:
        result = await tool.execute(operation="query", query="test")
        print(f"查询结果: {result}")
    except Exception as e:
        print(f"查询失败: {str(e)}")

    print("\n测试添加观察...")
    try:
        observation = {
            "entityName": "test_entity",
            "contents": ["这是一个新的观察"]
        }
        result = await tool.execute(operation="add_observations", observations=[observation])
        print(f"添加观察结果: {result}")
    except Exception as e:
        print(f"添加观察失败: {str(e)}")

    print("\n测试读取整个图谱...")
    try:
        result = await tool.execute(operation="read_graph")
        print(f"读取图谱结果: {result}")
    except Exception as e:
        print(f"读取图谱失败: {str(e)}")


# 如果直接运行此文件，则执行测试
# if __name__ == "__main__":
#     import asyncio
#     asyncio.run(test_knowledge_graph_tool())
