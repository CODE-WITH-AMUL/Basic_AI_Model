from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch
import json

# Initialize the AI model (you can move this to services.py if preferred)
chat_ai = None

def load_chat_model():
    global chat_ai
    if chat_ai is None:
        model_name = "microsoft/DialoGPT-medium"  # Or your preferred model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        chat_ai = pipeline("text-generation", model=model, tokenizer=tokenizer)

@login_required
def chat_view(request):
    """Render the main chat interface"""
    return render(request, 'index.html')  # Your single HTML file

@csrf_exempt
@login_required
def get_ai_response(request):
    """Handle AJAX requests for AI responses"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            user_input = data.get('message', '')
            
            # Load model if not loaded
            load_chat_model()
            
            # Generate response (adjust parameters as needed)
            response = chat_ai(
                user_input,
                max_length=100,
                num_return_sequences=1,
                pad_token_id=chat_ai.tokenizer.eos_token_id,
                no_repeat_ngram_size=3,
                do_sample=True,
                top_k=50,
                top_p=0.9,
                temperature=0.7
            )
            
            return JsonResponse({
                'response': response[0]['generated_text'].replace(user_input, '').strip()
            })
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Invalid request'}, status=400)

@login_required
def new_chat(request):
    """Handle request to start a new chat"""
    # In a real app, you might want to save the previous conversation
    return JsonResponse({'status': 'New chat started'})