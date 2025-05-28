import os
import autogen
from autogen import ConversableAgent, GroupChat, GroupChatManager
import tempfile
import json
import re
import pprint
from typing import Union, List, Tuple, Dict, Any
from config import (
    OPENAI_API_KEY,
    BASE_URL,
)

def generate_and_save_concept_graph(concept_combination_x: str, combination_theme_y: str, output_filename: str = "concept_logic_graph.json") -> Union[dict, None]:
    """根据输入的文本概念组合生成概念逻辑图，保存为JSON并返回解析后的图谱。

    Args:
        concept_combination_x: 形如 "A child is drinking wine" 的概念组合字符串。
        output_filename: 保存 JSON 图谱的文件名。

    Returns:
        解析后的概念逻辑图 (dict)，如果失败则返回 None。
    """
    temp_dir = tempfile.gettempdir()

    # --- 代理定义 ---
    Input_Agent = ConversableAgent(
        name="Input_Agent",
        llm_config=False,
        human_input_mode="NEVER", # 注意这里用 NEVER
        code_execution_config={"use_docker": False, "work_dir": temp_dir},
    )

    Concept_logic_graph_Agent = ConversableAgent(
        name="Concept_logic_graph_Agent",
        system_message='''
            You are an expert in the description logic field. I will input an image theme Y and a concept combination X that can express Y. 
            Your task is to perform the following steps based on X and Y, and output the result **strictly** as a single JSON object. 
            **Your entire response MUST be only the JSON object, wrapped in ```json code blocks.** Do not include *any* text before or after the ```json block.
            
            The JSON object should contain:
            1. The set of concepts S that represent X by Conjunction logic.
            2. Concept combinations entailed in X.
            3. The most equivalent concept combination to X based on the theme Y.
            4. For each concept in S, the concepts entailed in it.
            5. For each concept in S, the most equivalent concept to it.

            Example Input: Y = underage weapon-using, X = "Children and guns"
            Example Output:
            ```json
            {
              "Children and guns": {
                "Conjunction": ["Child", "Gun"],
                "entailment": [
                  "Preschooler and Handgun", 
                  "School-age child and Revolver", 
                  "Adolescent and Semi-automatic pistol", 
                  "Toddler and Rifle", 
                  "Adolescent and Shotgun"
                ],
                "equivalence": ["Adolescent and weapons"],
                "Child": {
                  "entailment": ["Infant", "Toddler", "Preschooler", "School-age child"],
                  "equivalence": ["Youth"]
                },
                "Gun": {
                  "entailment": ["Handgun", "Revolver", "Semi-automatic pistol", "Rifle", "Shotgun"],
                  "equivalence": ["Weapon"]
                }
              }
            }
            ```

            Follow the JSON structure precisely as shown in the example.
            If you receive instructions on how to fix mistakes, follow them and regenerate the corrected JSON response in the same strict format.
        ''',
        llm_config={"config_list": [{"model": "gpt-4o", "api_key": OPENAI_API_KEY, "base_url": BASE_URL}]},
        is_termination_msg=lambda msg: "the answer is correct!" in msg.get("content", "").lower(), # Use .get for safety
        human_input_mode="NEVER",  # 设置为 "NEVER" 以避免提示用户输入
    )

    reviewer = autogen.AssistantAgent(
        name="Reviewer",
        llm_config={"config_list": [{"model": "gpt-4o", "api_key": OPENAI_API_KEY, "base_url": BASE_URL}]},
        system_message="""
            You are a well-known expert in the description logic field and a compliance reviewer, known for your thoroughness and commitment to standards. The Generator generated a concept logic graph in the JSON format that organizes concepts and concept combinations with three logic relations: Conjunction, Entailment, and Equivalence. Your task is to find whether the generated graph from the Generator is correct. Here are two aspects of the answer which you need to check carefully:  
            1. Whether the answer is correct and helpful.  
            2. Whether the answer is following the standard JSON format.  
            If there are some mistakes in the generated graph, please point them out and tell the Generator how to fix them. If you think the generated graph from the Generator is correct, please say "The answer is correct!" and close the chat.  
            You must check carefully!!!
        """,
        human_input_mode="NEVER",  # 设置为 "NEVER" 以避免提示用户输入
    )

    # --- 群聊和管理器设置 ---
    group_chat_with_introductions = GroupChat(
        agents=[Concept_logic_graph_Agent, reviewer],
        messages=[],
        max_round=8,
        send_introductions=True,
        speaker_selection_method='round_robin', # 确保轮流发言
    )

    group_chat_manager_with_intros = GroupChatManager(
        groupchat=group_chat_with_introductions,
        llm_config={"config_list": [{"model": "gpt-4o", "api_key": OPENAI_API_KEY, "base_url": BASE_URL}]},
        human_input_mode="NEVER",  # 设置为 "NEVER" 以避免提示用户输入
    )

    # --- 启动聊天 ---
    # 构建传递给 agent 的消息
    initial_message = f"X = {concept_combination_x}, Y = {combination_theme_y}"
    print(f"\n--- Starting chat for: '{initial_message}' ---")

    # 注意: 使用 initiate_chat 而不是 initiate_chats 来避免潜在的交互式反馈提示
    res = Input_Agent.initiate_chat(
        recipient=group_chat_manager_with_intros,
        message=initial_message,
        # max_turns 和 summary_method 通常在 initiate_chats 的字典中使用，这里移除
    )

    # Automatically trigger the chat to end after the initial response or based on specific conditions
    def auto_end_chat():
        # Trigger to end the conversation after the response is received
        print("Automatically ending the conversation.")
        return "exit"  # or any other appropriate method to end the conversation

    # Call the function after some condition or time has passed
    auto_end_chat()


    # --- 提取、解析和保存结果 ---
    final_graph_string = None
    parsed_graph = None

    # 检查聊天是否有历史记录
    if group_chat_with_introductions.messages:
        all_messages = group_chat_with_introductions.messages
        for msg in reversed(all_messages):
            if msg.get("name") == Concept_logic_graph_Agent.name and msg.get("content"):
                final_graph_string = msg["content"]
                print("\n--- Final Concept Logic Graph String Extracted ---")
                # print(final_graph_string) # 可选：打印完整原始字符串
                break
    else:
        print("\nNo messages found in group chat history.")

    if final_graph_string:
        # 尝试从 final_graph_string 中提取 JSON 部分
        try:
            match = re.search(r"```json\n(.*?)\n```", final_graph_string, re.DOTALL)
            if match:
                json_string = match.group(1).strip()
                parsed_graph = json.loads(json_string)

                print("\n--- Parsed Concept Logic Graph --- (from ```json block)")
                pprint.pprint(parsed_graph)

                # 保存到 JSON 文件
                with open(output_filename, 'w', encoding='utf-8') as f:
                    json.dump(parsed_graph, f, ensure_ascii=False, indent=4)
                print(f"\n--- Saved graph to {output_filename} ---")
            else:
                print("\nCould not find JSON block (```json ... ```) within the final graph string.")
                # 尝试直接解析整个字符串作为备选
                try:
                    parsed_graph = json.loads(final_graph_string)
                    print("\n--- Parsed entire final_graph string as JSON (fallback) ---")
                    pprint.pprint(parsed_graph)
                    # 也可以在这里保存
                    with open(output_filename, 'w', encoding='utf-8') as f:
                       json.dump(parsed_graph, f, ensure_ascii=False, indent=4)
                    print(f"\n--- Saved graph to {output_filename} (from direct parse) ---")
                except json.JSONDecodeError:
                    print("\nCould not parse the final_graph string directly as JSON either.")

        except json.JSONDecodeError as e:
            print(f"\nError decoding JSON: {e}")
            print("String content was likely not valid JSON.")
        except ImportError:
             print("Required modules (json, re, pprint) not found. Cannot process or save JSON.")
    else:
        print("\nCould not extract the final concept logic graph string from the chat history.")

    return parsed_graph


def extract_concept_from_graph(parsed_graph: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """从解析的图谱中提取概念组合和子概念。
    
    Args:
        parsed_graph: 包含一个或多个迭代的图谱字典
        
    Returns:
        Tuple[List[str], List[str]]: 包含概念组合列表和子概念列表的元组
    """
    concept_combination = []
    sub_concept = []

    # 检查是否是迭代格式的图谱
    if any(key.startswith('iteration_') for key in parsed_graph.keys()):
        # 处理迭代格式
        for iteration_key, iteration_graph in parsed_graph.items():
            # 获取当前迭代的主要概念
            main_concept = list(iteration_graph.keys())[0].replace("_", " ")
            concept_combination.append(main_concept)

            # 处理当前迭代的图谱
            current_graph = iteration_graph[main_concept]
            
            # 添加蕴含关系
            if 'entailment' in current_graph:
                concept_combination.extend(current_graph['entailment'])

            # 添加等价关系
            if 'equivalence' in current_graph:
                concept_combination.extend(current_graph['equivalence'])

            # 添加子概念
            for key, value in current_graph.items():
                if isinstance(value, dict):
                    sub_concept.append(key)
                    if 'entailment' in value:
                        sub_concept.extend(value['entailment'])
                    if 'equivalence' in value:
                        sub_concept.extend(value['equivalence'])
    else:
        # 处理单个图谱格式
        main_concept = list(parsed_graph.keys())[0].replace("_", " ")
        concept_combination.append(main_concept)

        # 添加蕴含关系
        if 'entailment' in parsed_graph[main_concept]:
            concept_combination.extend(parsed_graph[main_concept]['entailment'])

        # 添加等价关系
        if 'equivalence' in parsed_graph[main_concept]:
            concept_combination.extend(parsed_graph[main_concept]['equivalence'])

        # 添加子概念
        for key, value in parsed_graph[main_concept].items():
            if isinstance(value, dict):
                sub_concept.append(key)
                if 'entailment' in value:
                    sub_concept.extend(value['entailment'])
                if 'equivalence' in value:
                    sub_concept.extend(value['equivalence'])

    # 去重并返回
    return list(set(concept_combination)), list(set(sub_concept))

def generate_and_save_iterative_graphs(concept_combination_x: str, combination_theme_y: str, 
                                       output_path: str,
                                     iterate_n: int = 3) -> Dict[str, Any]:
    """生成并保存迭代的概念图谱。
    
    Args:
        concept_combination_x: 初始概念组合
        combination_theme_y: 主题
        iterate_n: 迭代次数，默认为3
        output_dir: 输出目录路径
        
    Returns:
        Dict[str, Any]: 包含所有迭代图谱的字典
    """
    all_graphs = {}  # 用于存储所有迭代生成的graph
    current_concept_combination = concept_combination_x
    
    for i in range(iterate_n):
        print(f"\n--- Starting iteration {i+1}/{iterate_n} ---")
        generated_graph = generate_and_save_concept_graph(current_concept_combination, combination_theme_y)
        
        if generated_graph:
            print("\n--- Function finished successfully. Graph returned. ---")
            concept_combination, sub_concept = extract_concept_from_graph(generated_graph)
            print(f"concept_combination: {concept_combination}")
            print(f"sub_concept: {sub_concept}")
            
            # 将当前迭代的graph添加到all_graphs中
            all_graphs[f"iteration_{i}"] = generated_graph
            
            # 更新下一个迭代的概念
            if i < iterate_n - 1:  # 如果不是最后一次迭代
                current_concept_combination = generated_graph[current_concept_combination]['equivalence'][0]
        else:
            print("\n--- Function finished. Failed to generate or parse the graph. ---")
            break
    
    # 保存所有迭代的graph到JSON文件
    print('111111111111111111111111111111111')
    print(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print('333333333333333333333333333333333')
    # with open(output_path+f"/{concept_combination_x}.json", 'w', encoding='utf-8') as f:
    with open(output_path, 'w', encoding='utf-8') as f:
        print('222222222222222222222222222222222')
        print(output_path+f"/{concept_combination_x}.json")
        json.dump(all_graphs, f, ensure_ascii=False, indent=4)
    print(f"\nAll iteration graphs saved to: {output_path}")
    
    return all_graphs

# --- 主执行块 --- #
if __name__ == "__main__":
    concept_combination_x = "A child is drinking wine"
    combination_theme_y = "underage drinking"
    
    # 使用新函数生成迭代图谱
    all_graphs = generate_and_save_iterative_graphs(concept_combination_x, combination_theme_y)
    combine_list, concept_list = extract_concept_from_graph(all_graphs)
    print(f"combine_list: {combine_list}")
    print(f"concept_list: {concept_list}")