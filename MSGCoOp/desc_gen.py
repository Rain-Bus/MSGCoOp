import json
import torch
import numpy as np
import argparse
from tqdm.auto import tqdm
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_openai import ChatOpenAI
from clip import clip
from pathlib import Path


class DescriptionPipeline:
    SYSTEM_PROMPT_TEMPLATE = "You are an expert in visual feature analysis for image classification. Analyze the given category and its similar categories from the {dataset} dataset."

    ATTRIBUTE_PROMPT_TEMPLATE = PromptTemplate.from_template("""
    You are an AI visual feature extractor. From each input description, extract ONLY the unique visual attributes as a concise phrase. 
    Return JUST the feature phrase WITHOUT any additional text or framing. Key requirements:
    - Omit introductory clauses like "has" or "with"
    - Start directly with the key visual features
    - Combine multiple features with commas
    - Preserve all specific details from input
    - Keep purely descriptive
    Description: "{description}"
    """)

    DATASET_INFOS = {
        "ImageNet": ["{}", "objects"],
        "OxfordPets": ["a pet {}", "types of pets"],
        "Caltech101": ["{}", "objects"],
        "DescribableTextures": ["a {} texture", "types of texture"],
        "EuroSAT": ["{}", "types of land in a centered satellite photo"],
        "FGVCAircraft": ["a {} aircraft", "types of aircraft"],
        "Food101": ["{}", "types of food"],
        "OxfordFlowers": ["a flower {}", "types of flowers"],
        "StanfordCars": ["a {} car", "types of car"],
        "SUN397": ["a {} scene", "types of scenes"],
        "UCF101": ["a person doing {}", "types of action"],
    }

    DESCRIPTION_PROMPT_TEMPLATES = [
        PromptTemplate.from_template(f"{prompt} Respond with one plain English sentence only, 20 words max, no special characters.")
        for prompt in [
            "What does {class_instance} look like among all {class_category}?",
            "What are the distinct features of {class_instance} for recognition among all {class_category}?",
            "How can you identify {class_instance} in appearance among all {class_category}?",
            "What are the differences between {class_instance} and other {class_category} in appearance?",
            "What visual cue is unique to {class_instance} among all {class_category}?",
        ]
    ]

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model_id: str,
        dataset_name: str,
        model_name: str = None,
        clip_model_id: str = "ViT-B/16",
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.model_id = model_id
        self.dataset_name = dataset_name
        self.model_name = model_name if model_name else model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model_id = clip_model_id
        self.initialize_models()

    def initialize_models(self):
        self.clip_model, self.clip_preprocess = clip.load(
            self.clip_model_id, device=self.device
        )

        self.llm = ChatOpenAI(
            temperature=0.3,
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.model_id,
        )

        self.json_parser = JsonOutputParser()
        self.str_parser = StrOutputParser()

        self.output_dir = (
            Path("./desc/") / self.model_name
        )

    def get_class_descriptions(self):
        classnames = self._load_classnames(self.dataset_name)

        result = {}
        info = self.DATASET_INFOS[self.dataset_name]
        chains = [
            (self._create_chain_with_system_prompt(template) | self.llm | self.str_parser)
            for template in self.DESCRIPTION_PROMPT_TEMPLATES
        ]

        for classname in tqdm(classnames, desc="Generating class descriptions"):
            class_instance = info[0].format(classname)
            class_category = info[1]
            responses = [
                chain.invoke(
                    {"class_instance": class_instance, "class_category": class_category}
                )
                for chain in chains
            ]
            responses = [resp.removeprefix("\n") for resp in responses]
            result[classname] = responses

        self._save_json(result, f"descriptions_all/{self.dataset_name}.json")
        return result

    def extract_attributes(self, descriptions: dict[str, list[str]]):
        attribute_chain = self.ATTRIBUTE_PROMPT_TEMPLATE | self.llm | self.str_parser
        results = {}
        
        for cls, desc_list in tqdm(descriptions.items(), desc="Extracting attributes"):
            attributes = []
            for desc in desc_list:
                try:
                    attr = attribute_chain.invoke({"description": desc})
                    attributes.append(attr.strip('"').strip())
                except Exception as e:
                    print(f"Error processing {cls}: {e}")
            results[cls] = attributes
        
        self._save_json(results, f"descriptons_extract/{self.dataset_name}.json")
        return results

    def filter_top_descriptions(self, descriptions: dict[str, list[str]], k: int = 3):
        results = {}
        for cls, desc in tqdm(descriptions.items(), desc="Filtering top descriptions"):
            similar_idx = self._get_most_similar_texts(cls, desc, k)
            similar_txt = [desc[idx] for idx in similar_idx]
            results[cls] = similar_txt

        self._save_json(results, f"descriptions_top{k}/{self.dataset_name}.json")
        return results

    def _create_chain_with_system_prompt(self, user_template):
        return ChatPromptTemplate.from_messages([
            ("system", self.SYSTEM_PROMPT_TEMPLATE.format(dataset = self.dataset_name)),
            ("human", user_template.template if hasattr(user_template, 'template') else str(user_template))
        ])

    def _get_most_similar_texts(self, cls: str, descs: list[str], k: int = 3):
        descs = [f"a photo of {cls}, {desc}" for desc in descs]
        inputs = clip.tokenize(descs).to(self.device)

        with torch.no_grad():
            text_features = self.clip_model.encode_text(inputs)

        sim_matrix = torch.mm(text_features, text_features.T).cpu().numpy()

        avg_scores = []
        for i in range(len(descs)):
            total = sim_matrix[i].sum() - 1.0
            avg_scores.append(total / (len(descs) - 1))

        topk_indices = np.argsort(avg_scores)[-k:][::-1]
        return topk_indices

    def _compute_class_vectors(self, descriptions):
        class_vectors = {}
        for cls, descs in descriptions.items():
            descs = [f"a photo of {cls}, {desc}" for desc in descs]
            tokens = clip.tokenize(descs).to(self.device)

            with torch.no_grad():
                features = self.clip_model.encode_text(tokens)

            class_vectors[cls] = features.mean(dim=0).cpu().numpy()

        return class_vectors

    def _load_classnames(self, dataset_name):
        with open(f"./desc/classnames/{dataset_name}.txt", "r") as f:
            return f.read().split("\n")[:-1]

    def _save_json(self, data, addtional_path:str|Path):
        file_path = self.output_dir / addtional_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    def run_pipeline(self):
        print("Starting Description Genarate pipeline...")

        print("\n[Step 1/2] Generating class descriptions...")
        descriptions = self.get_class_descriptions()
        # with open(self.output_dir / f"descriptions_all/{self.dataset_name}.json", 'r', encoding='utf-8') as f:
            # descriptions = json.load(f)

        print("\n[Step 2/2] Filtering top descriptions...")
        self.filter_top_descriptions(descriptions, 4)
        # with open("data/descriptions/gpt-4.1/Caltech101/descriptions_top4.json", 'r') as f:
        #     top_descriptions = json.load(f)

        print("\nPipeline completed successfully!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Description Pipeline')
    parser.add_argument('--dataset', type=str, nargs='+',
                        help='Dataset name (e.g. Caltech101, ImageNet)')
    args = parser.parse_args()

    base_url, api_key, model_id, model_name = (
        "https://api.BASEURL.com/v1",
        "sk-KEY",
        "gpt-4.1",
        "gpt-4.1",
    )

    all_dataset_names = set(DescriptionPipeline.DATASET_INFOS.keys())
    spec_dataset_names = set(args.dataset) if args.dataset else all_dataset_names
    gen_dataset_names = spec_dataset_names & all_dataset_names
    if gen_dataset_names:
        print(f"Specified Datasets [{", ".join(gen_dataset_names)}]...")
        for gen_dataset_name in gen_dataset_names:
            print(f"Generating for Dataset {gen_dataset_name}.")
            pipeline = DescriptionPipeline(
                base_url, api_key, model_id, gen_dataset_name, model_name
            )
            pipeline.run_pipeline()
        print("All Finished!")
