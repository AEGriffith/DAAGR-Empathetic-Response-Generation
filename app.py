import gradio as gr
import tensorflow as tf
import time
import warnings
import os
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, AutoModelForCausalLM, \
    TFAutoModelForQuestionAnswering, pipeline, TFT5ForConditionalGeneration

# Warning Suppression
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
tf.get_logger().setLevel('ERROR')

emotion_tokenizer = AutoTokenizer.from_pretrained("aegrif/CIS6930_DAAGR_Classification")
emotion_model = TFAutoModelForSequenceClassification.from_pretrained("aegrif/CIS6930_DAAGR_Classification")

# generation models
# encoder
distilbert_model = TFAutoModelForQuestionAnswering.from_pretrained("aegrif/CIS6930_DAAGR_DistilBert")
distilbert_tokenizer = AutoTokenizer.from_pretrained("aegrif/CIS6930_DAAGR_DistilBert")

# decoder
gpt2_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
gpt2_tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")

# encoder-decoder
t5_model = TFT5ForConditionalGeneration.from_pretrained("t5-small")
t5_tokenizer = AutoTokenizer.from_pretrained("t5-small")

emotion_dict = {'disappointed': 0, 'annoyed': 1, 'excited': 2, 'afraid': 3, 'disgusted': 4, 'grateful': 5,
                'impressed': 6, 'prepared': 7}
inverted_emotion_dict = {v: k for k, v in emotion_dict.items()}


def get_context(user_input):
    new_user_input_ids = emotion_tokenizer.encode(user_input, return_tensors='tf')
    output = emotion_model.predict(new_user_input_ids)[0]
    prediction = tf.argmax(output, axis=1).numpy()[0]
    context = inverted_emotion_dict.get(prediction)

    return context


def predict_distilbert(user_input, history):
    # Get the context from the user input
    context = get_context(user_input)

    # Generate a response using the DistilBert model
    question_answerer = pipeline("question-answering", model=distilbert_model, tokenizer=distilbert_tokenizer)
    output = question_answerer(question=user_input, context=context)
    # Decode the generated response
    bot_response = output['answer']

    return bot_response


def predict_gpt2(user_input, history):
    # Get the context from the user input
    context = get_context(user_input)

    # Generate a response using the DialoGPT model
    chatbot = pipeline("text-generation", model=gpt2_model, tokenizer=gpt2_tokenizer, pad_token_id=50256)
    output = chatbot(user_input + context, max_length=100, num_return_sequences=1)
    # Decode the generated response
    bot_response = output[0]['generated_text']

    return bot_response


def predict_t5(user_input, history):
    # Get the context from the user input
    context = get_context(user_input)

    # Generate a response using the T5 model
    chatbot = pipeline("text2text-generation", model=t5_model, tokenizer=t5_tokenizer, pad_token_id=50256)
    bot_response = chatbot(user_input + context, max_length=100, num_return_sequences=1)[0]['generated_text']

    return bot_response


def user(user_message, history):
    return "", history + [[user_message, None]]


def distil_bot(history):
    user_message = history[-1][0]
    bot_message = predict_distilbert(user_message, history)
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
        distil_bot, chatbot1, chatbot1
    )
    msg2.submit(user, [msg2, chatbot2], [msg2, chatbot2], queue=False).then(
        gpt2_bot, chatbot2, chatbot2
    )
    msg3.submit(user, [msg3, chatbot3], [msg3, chatbot3], queue=False).then(
        t5_bot, chatbot3, chatbot3
    )

if __name__ == "__main__":
    demo.launch()
