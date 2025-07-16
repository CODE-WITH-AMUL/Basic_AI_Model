

# Create your views here.
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch

class ChatAIService:
    def __init__(self):
        self.model_name = "microsoft/DialoGPT-medium"  # Or other model
        self.model = None
        self.tokenizer = None
        self.chat_history = []
        
    def load_model(self):
        """Load the pretrained model"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        
    def generate_response(self, user_input):
        """Generate AI response"""
        if not self.model:
            self.load_model()
            
        # Encode the new user input
        new_user_input_ids = self.tokenizer.encode(
            user_input + self.tokenizer.eos_token,
            return_tensors='pt'
        )

        # Append to chat history
        self.chat_history.append(new_user_input_ids)
        
        # Generate response
        bot_input_ids = torch.cat(self.chat_history, dim=-1) if len(self.chat_history) > 1 else new_user_input_ids
        chat_history_ids = self.model.generate(
            bot_input_ids,
            max_length=1000,
            pad_token_id=self.tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
            do_sample=True,
            top_k=100,
            top_p=0.7,
            temperature=0.8
        )
        
        # Decode and return response
        response = self.tokenizer.decode(
            chat_history_ids[:, bot_input_ids.shape[-1]:][0],
            skip_special_tokens=True
        )
        
        # Store the conversation
        self.chat_history.append(chat_history_ids)
        
        return response

# Singleton instance
chat_service = ChatAIService()