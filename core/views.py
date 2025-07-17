from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json

# Conversation history
conversation_history = []

def load_model():
    model_name = "tiiuae/falcon-7b-instruct" # Using Gemma 2B
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16
    )
    return model, tokenizer

model, tokenizer = load_model()

@csrf_exempt
def get_ai_response(request):
    global conversation_history
    
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            user_input = data.get('message', '')
            
            # Format the prompt with conversation history
            prompt = format_prompt(conversation_history, user_input)
            
            # Generate response
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
            
            # Decode and clean response
            full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            ai_response = extract_ai_response(full_response, prompt)
            
            # Update conversation history
            conversation_history.append({"user": user_input, "ai": ai_response})
            
            return JsonResponse({'response': ai_response})
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Invalid request'}, status=400)

def format_prompt(history, new_input):
    """Format the prompt with conversation history"""
    prompt = "You are a helpful AI assistant. Answer questions clearly and concisely.\n\n"
    
    for turn in history[-4:]:  # Keep last 4 exchanges for context
        prompt += f"User: {turn['user']}\n"
        prompt += f"Assistant: {turn['ai']}\n"
    
    prompt += f"User: {new_input}\n"
    prompt += "Assistant:"
    
    return prompt

def extract_ai_response(full_text, prompt):
    """Extract just the AI's response from the full generated text"""
    return full_text[len(prompt):].split("User:")[0].strip()