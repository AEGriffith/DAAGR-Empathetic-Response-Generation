import gradio as gr
import tensorflow as tf
import time
import warnings
import os
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, AutoModelForCausalLM

# Warning Suppression
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
tf.get_logger().setLevel('ERROR')

emotion_tokenizer = AutoTokenizer.from_pretrained("aegrif/CIS6930_DAAGR_Classification")
emotion_model = TFAutoModelForSequenceClassification.from_pretrained("aegrif/CIS6930_DAAGR_Classification")

#
tokenizer1 = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
model1 = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

# tokenizer2 = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
# model2 = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
#
# tokenizer3 = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
# model3 = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")

emotion_dict = {'disappointed': 0, 'annoyed': 1, 'excited': 2, 'afraid': 3, 'disgusted': 4, 'grateful': 5,
                'impressed': 6, 'prepared': 7}
inverted_emotion_dict = {v: k for k, v in emotion_dict.items()}


def get_context(user_input):
    new_user_input_ids = emotion_tokenizer.encode(user_input, return_tensors='tf')
    output = emotion_model.predict(new_user_input_ids)[0]
    prediction = tf.argmax(output, axis=1).numpy()[0]
    context = inverted_emotion_dict.get(prediction)

    return context


def predict(user_input, history):
    # Get the context from the user input
    # NOTE: Not implemented yet
    context = get_context(user_input)

    # Combine the conversation history with the user input (without emotion context and labels) and add the EOS token after each message
    conversation = "".join([f"{msg[0]}{tokenizer1.eos_token}{msg[1]}{tokenizer1.eos_token}" for msg in history if
                            msg[1] is not None]) + f" {user_input} {tokenizer1.eos_token}"

    # Tokenize the conversation
    input_ids = tokenizer1.encode(conversation, return_tensors="pt")

    # Generate a response using the DialoGPT model
    output = model1.generate(
        input_ids,
        max_length=2048,
        pad_token_id=tokenizer1.eos_token_id,
        no_repeat_ngram_size=3,
        temperature=0.8,  # Adjust the temperature for more randomness in the generated text
        top_k=50,  # Set the top_k parameter for controlling the token selection
        top_p=0.95,  # Set the top_p parameter for controlling the token selection
    )

    # Decode the generated response
    bot_response = tokenizer1.decode(output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

    return bot_response


def user(user_message, history):
    return "", history + [[user_message, None]]


def bot(history):
    user_message = history[-1][0]
    bot_message = predict(user_message, history)
    history[-1][1] = bot_message
    time.sleep(1)
    return history


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            chatbot1 = gr.Chatbot().style()
            state1 = gr.State([])
            msg1 = gr.Textbox(show_label=False, placeholder="Enter text and press enter").style(container=False)

    msg1.submit(user, [msg1, chatbot1], [msg1, chatbot1], queue=False).then(
        bot, chatbot1, chatbot1
    )

if __name__ == "__main__":
    demo.launch()
