from FlagEmbedding import BGEM3FlagModel
from chromadb import Documents, EmbeddingFunction, Embeddings
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


class CustomEmbeddingFunction(EmbeddingFunction):

    def __init__(self):
        self.model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

    def __call__(self, input: Documents) -> Embeddings:
        embeddings = self.model.encode(input, 
                             batch_size=12, 
                             max_length=1024,
                            )['dense_vecs']
        return embeddings


class Model:

    def __init__(self, model_path="microsoft/Phi-4-mini-instruct"):

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=False,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )
 
        self.generation_args = {
            "max_new_tokens": 500,
            "return_full_text": False,
            "do_sample": False,
        }
    
    def generate(self, query, rag_input=None):
        messages = self.construct_prompt(query, rag_input)
        response = self.pipe(messages, **self.generation_args)[0]['generated_text']
        return response

    def construct_prompt(self, query, rag_input):
        if rag_input:
            messages = [
                {
                    "role": "system", 
                    "content": (
                        "You are a helpful AI assistant that knows everything about cars and how to fix them."
                        "Give no cautions, except those that are in the manual."
                        "Use only the relevant information."
                        f"Answer user's question given the following context from the car's manual: {rag_input}"
                    )
                },
                {
                    "role": "user", 
                    "content": query
                },
            ]
        else:
            messages = [
                {
                    "role": "system", 
                    "content": (
                        "You are a helpful AI assistant that knows everything about cars and how to fix them."
                    )
                },
                {
                    "role": "user", 
                    "content": query
                },
            ]
        return messages
