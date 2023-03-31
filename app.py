import gradio as gr
import tensorflow as tf
import time
import warnings
import os
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)

# Warning Suppression
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# warnings.filterwarnings('ignore', category=DeprecationWarning)
# warnings.filterwarnings('ignore', category=FutureWarning)
# warnings.filterwarnings('ignore', category=UserWarning)
# tf.get_logger().setLevel('ERROR')

# emotion classification
emotion_tokenizer = "aegrif/CIS6930_DAAGR_Classification"
emotion_model = "aegrif/CIS6930_DAAGR_Classification"
emotion_pipeline = pipeline("text-classification", model=emotion_model, tokenizer=emotion_tokenizer)

# generation models

# no emotion
gpt2_model_no_emo = "aegrif/CIS6930_DAAGR_GPT2_NoEmo"
gpt2_tokenizer_no_emo = "aegrif/CIS6930_DAAGR_GPT2_NoEmo"
chatbot_gpt_no_emo = pipeline(model=gpt2_model_no_emo, tokenizer=gpt2_tokenizer_no_emo, pad_token_id=50256)


# decoder
gpt2_model_emo = "aegrif/CIS6930_DAAGR_GPT2_Emo"
gpt2_tokenizer = "aegrif/CIS6930_DAAGR_GPT2_Emo"
chatbot_gpt_emo = pipeline(model=gpt2_model_emo, tokenizer=gpt2_tokenizer, pad_token_id=50256)


# encoder-decoder
t5_model_emo = "aegrif/CIS6930_DAAGR_T5_Emo"
t5_tokenizer = "t5-small"
chatbot_t5_emo = pipeline(model=t5_model_emo, tokenizer=t5_tokenizer)

emotion_dict = {'disappointed': 0, 'annoyed': 1, 'excited': 2, 'afraid': 3, 'disgusted': 4, 'grateful': 5,
                'impressed': 6, 'prepared': 7}
inverted_emotion_dict = {v: k for k, v in emotion_dict.items()}


def get_context(user_input):

    output = emotion_pipeline(user_input)[0]['label']

    context = inverted_emotion_dict.get(int(output[-1]))

    return context


def predict_gpt2_no_emo(user_input, history):
    user_input = "<|user|>" + user_input + " <|bot|>"

    output = chatbot_gpt_no_emo(
        user_input,
        max_new_tokens=40,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.5,
        renormalize_logits=True,
        exponential_decay_length_penalty=(5, 1.1),
        no_repeat_ngram_size=3,
        repetition_penalty=1.5
    )

    # Decode the generated response
    bot_response = output[0]['generated_text'].split("<|bot|>")[1].strip()

    return bot_response


def predict_gpt2(user_input, history):
    # Get the context from the user input
    context = get_context(user_input)

    user_input = f"<|context|>{context} <|user|>{user_input} <|bot|>"
    # Generate a response using the DialoGPT model

    output = chatbot_gpt_emo(
        user_input,
        max_new_tokens=40,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.5,
        renormalize_logits=True,
        exponential_decay_length_penalty=(5, 1.1),
        no_repeat_ngram_size=3,
        repetition_penalty=1.5
    )

    # Decode the generated response
    bot_response = output[0]['generated_text'].split("<|bot|>")[1].strip()

    return bot_response


def predict_t5(user_input, history):
    # Get the context from the user input
    context = get_context(user_input)

    user_input = f"question: {user_input} context: {context} </s>"
    # Generate a response using the T5 model
    bot_response = chatbot_t5_emo(
        user_input,
        max_new_tokens=40,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.5,
        renormalize_logits=True,
        exponential_decay_length_penalty=(5, 1),
        no_repeat_ngram_size=3,
        repetition_penalty=1.5
    )[0]['generated_text']

    return bot_response


def user(user_message, history):
    return "", history + [[user_message, None]]


def gpt2_bot_no_emo(history):
    user_message = history[-1][0]
    bot_message = predict_gpt2_no_emo(user_message, history)
    history[-1][1] = bot_message
    time.sleep(1)
    return history

def gpt2_bot(history):
    user_message = history[-1][0]
    bot_message = predict_gpt2(user_message, history)
    history[-1][1] = bot_message
    time.sleep(1)
    return history


def t5_bot(history):
    user_message = history[-1][0]
    bot_message = predict_t5(user_message, history)
    history[-1][1] = bot_message
    time.sleep(1)
    return history


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            chatbot1 = gr.Chatbot().style()
            msg1 = gr.Textbox(show_label=False, placeholder="Enter text and press enter").style(container=False)
        with gr.Column():
            chatbot2 = gr.Chatbot().style()
            msg2 = gr.Textbox(show_label=False, placeholder="Enter text and press enter").style(container=False)
        with gr.Column():
            chatbot3 = gr.Chatbot().style()
            msg3 = gr.Textbox(show_label=False, placeholder="Enter text and press enter").style(container=False)

    msg1.submit(user, [msg1, chatbot1], [msg1, chatbot1], queue=False).then(
        gpt2_bot_no_emo, chatbot1, chatbot1
    )
    msg2.submit(user, [msg2, chatbot2], [msg2, chatbot2], queue=False).then(
        gpt2_bot, chatbot2, chatbot2
    )
    msg3.submit(user, [msg3, chatbot3], [msg3, chatbot3], queue=False).then(
        t5_bot, chatbot3, chatbot3
    )

if __name__ == "__main__":
    demo.launch()
