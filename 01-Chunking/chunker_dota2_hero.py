# HERO CHUNKING

import os
import json
from transformers import AutoTokenizer, logging
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Suppress Hugging Face warnings
logging.set_verbosity_error()

TOKENIZER_NAME = "BAAI/bge-base-en-v1.5"
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

def get_token_count(text):
    return len(tokenizer.encode(text, add_special_tokens=True))

SAFE_NODES = ['basics', 'changelog']
LIST_NODES = ['abilities', 'innate_abilities', 'invoked_abilities']
SPLIT_NODES = ['trivia'] 
COMPLEX_NODES = ['guide'] 

def create_chunk(text_content, hero_name, primary_node, sub_node="root", chunk_index=0):
    full_path = f"{primary_node}.{sub_node}" if sub_node != "root" else primary_node
    prefix = f"Game: Dota 2 | Hero: {hero_name} | Path: {full_path} | Data: "
    final_text = prefix + str(text_content)
    
    metadata = {
        "game": "Dota 2",
        "hero": hero_name,
        "primary_node": primary_node,
        "sub_node": sub_node,
        "full_path": full_path,
        "chunk_index": chunk_index
    }
    
    return {
        "page_content": final_text,
        "metadata": metadata,
        "token_count": get_token_count(final_text)
    }

def process_hero_json(filepath, filename):
    final_chunks = []
    
    with open(filepath, 'r', encoding='utf-8') as file:
        data = json.load(file)
        hero_name = data.get('profile', {}).get('name', filename.replace('.json', ''))
        
        def chunk_with_safety(text_content, primary_node, sub_node):
            dummy_prefix = f"Game: Dota 2 | Hero: {hero_name} | Path: {primary_node}.{sub_node} | Data: "
            max_chunk_size = 512 - get_token_count(dummy_prefix) - 2
            
            if get_token_count(text_content) > max_chunk_size:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=max_chunk_size,
                    chunk_overlap=50,
                    length_function=get_token_count,
                    separators=["\n\n", "\n", "},", '",', ". ", " ", ""]
                )
                split_texts = text_splitter.split_text(text_content)
                for idx, split_text in enumerate(split_texts):
                    final_chunks.append(create_chunk(split_text, hero_name, primary_node, sub_node, idx))
            else:
                final_chunks.append(create_chunk(text_content, hero_name, primary_node, sub_node, 0))

        def process_complex_node(node_data, primary_node, current_sub_path):
            if isinstance(node_data, dict):
                for k, v in node_data.items():
                    new_path = f"{current_sub_path}.{k}" if current_sub_path else k
                    process_complex_node(v, primary_node, new_path)
            elif isinstance(node_data, list):
                if len(node_data) > 0 and isinstance(node_data[0], dict):
                    for i, item in enumerate(node_data):
                        title = item.get('title') or item.get('name') or f"item_{i}"
                        new_path = f"{current_sub_path}.{title}" if current_sub_path else title
                        text_content = json.dumps(item, ensure_ascii=False)
                        chunk_with_safety(text_content, primary_node, new_path)
                else:
                    text_content = json.dumps(node_data, ensure_ascii=False)
                    chunk_with_safety(text_content, primary_node, current_sub_path)
            else:
                chunk_with_safety(str(node_data), primary_node, current_sub_path)

        def bundle_and_chunk_array(array_data, primary_node, sub_node):
            if not isinstance(array_data, list) or not array_data:
                return

            dummy_prefix = f"Game: Dota 2 | Hero: {hero_name} | Path: {primary_node}.{sub_node} | Data: "
            max_chunk_size = 512 - get_token_count(dummy_prefix) - 2

            current_bundle = []
            current_tokens = 0
            local_index = 0

            for item in array_data:
                item_str = json.dumps(item, ensure_ascii=False) if isinstance(item, dict) else str(item).strip()
                item_tokens = get_token_count(item_str + " | ")

                if current_tokens + item_tokens > max_chunk_size and current_bundle:
                    bundle_str = " | ".join(current_bundle)
                    final_chunks.append(create_chunk(bundle_str, hero_name, primary_node, sub_node, local_index))
                    local_index += 1
                    current_bundle = [item_str]
                    current_tokens = item_tokens
                else:
                    current_bundle.append(item_str)
                    current_tokens += item_tokens

            if current_bundle:
                bundle_str = " | ".join(current_bundle)
                final_chunks.append(create_chunk(bundle_str, hero_name, primary_node, sub_node, local_index))

        def process_ability_semantically(ability_dict, primary_node):
            ability_name = ability_dict.get('name') or ability_dict.get('id') or "Unknown_Ability"
            
            core_dict = {k: v for k, v in ability_dict.items() if k != 'notes'}
            chunk_with_safety(json.dumps(core_dict, ensure_ascii=False), primary_node, f"{ability_name}.core")
                
            if 'notes' in ability_dict and isinstance(ability_dict['notes'], list):
                bundle_and_chunk_array(ability_dict['notes'], primary_node, f"{ability_name}.notes")

        for node_name, node_data in data.items():
            
            if node_name in SAFE_NODES:
                raw_str = json.dumps(node_data, ensure_ascii=False)
                cleaned_str = raw_str.replace("100/4001", "100/400")
                chunk_with_safety(cleaned_str, node_name, "root")
                
            elif node_name in LIST_NODES:
                if isinstance(node_data, list):
                    for item in node_data:
                        process_ability_semantically(item, node_name)
                else:
                    chunk_with_safety(json.dumps(node_data, ensure_ascii=False), node_name, "root")
                    
            elif node_name in SPLIT_NODES:
                chunk_with_safety(json.dumps(node_data, ensure_ascii=False), node_name, "root")
                
            elif node_name in COMPLEX_NODES:
                if isinstance(node_data, dict):
                    for category, content in node_data.items():
                        category = category.strip("'\"")
                        
                        if category == 'recommended_items' and isinstance(content, dict):
                            for phase, items_data in content.items():
                                formatted_items = []
                                iterable_items = items_data.values() if isinstance(items_data, dict) else items_data
                                
                                for item_val in iterable_items:
                                    if isinstance(item_val, dict) and 'item' in item_val:
                                        desc = item_val.get('description', '').strip()
                                        formatted_items.append(f"{item_val['item']}: {desc}")
                                
                                if formatted_items:
                                    bundle_and_chunk_array(formatted_items, f"{node_name}.recommended_items", phase)

                        elif category == 'tips' and isinstance(content, dict):
                            for tip_type, tips_data in content.items():
                                if isinstance(tips_data, list):
                                    bundle_and_chunk_array(tips_data, f"{node_name}.tips", tip_type)
                                elif isinstance(tips_data, dict):
                                    for sub_tip_type, sub_tips_list in tips_data.items():
                                        if isinstance(sub_tips_list, list):
                                            bundle_and_chunk_array(sub_tips_list, f"{node_name}.tips.{tip_type}", sub_tip_type)
                                        else:
                                            process_complex_node(sub_tips_list, f"{node_name}.tips.{tip_type}", sub_tip_type)
                                else:
                                    process_complex_node(tips_data, f"{node_name}.tips", tip_type)
                        
                        else:
                            process_complex_node(content, node_name, category)
                
            elif node_name == 'profile':
                profile_metadata = {}
                for sub_key, sub_value in node_data.items():
                    val_str = str(sub_value)
                    if isinstance(sub_value, str) and get_token_count(val_str) > 100:
                        chunk_with_safety(val_str, node_name, sub_key)
                    else:
                        profile_metadata[sub_key] = sub_value
                if profile_metadata:
                    chunk_with_safety(json.dumps(profile_metadata, ensure_ascii=False), node_name, "metadata")
                    
    return final_chunks

def run_production_pipeline(input_folder, output_filepath):
    print("Starting Final Production Pipeline...")
    all_chunks = []
    processed_count = 0
    
    for filename in os.listdir(input_folder):
        if filename.endswith('.json'):
            filepath = os.path.join(input_folder, filename)
            try:
                chunks = process_hero_json(filepath, filename)
                all_chunks.extend(chunks)
                processed_count += 1
                print(f"[{processed_count}] Processed {filename} -> {len(chunks)} chunks")
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                
    print(f"\nPipeline Complete! Total chunks generated across all {processed_count} heroes: {len(all_chunks)}")
    
    print(f"Writing data to {output_filepath}...")
    with open(output_filepath, 'w', encoding='utf-8') as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
            
    print("Success!")

if __name__ == "__main__":
    INPUT_FOLDER = r"your_path"
    OUTPUT_FILE = r"your_path"
    
    run_production_pipeline(INPUT_FOLDER, OUTPUT_FILE)