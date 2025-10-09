import numpy as np
import pandas as pd
from transformers import pipeline
import os

class AFDInfinityAMI:
    def __init__(self):
        self.llm = pipeline("text-generation", model="distilbert/distilgpt2")
        self.sentiment_analyzer = pipeline("sentiment-analysis")
        self.memory_file = 'data/response_log.csv'
        self.alpha, self.beta, self.gamma, self.delta = 1.0, 1.0, 0.5, 0.5
        if not os.path.exists(self.memory_file):
            try:
                pd.DataFrame(columns=['prompt', 'response', 'coherence']).to_csv(self.memory_file, index=False)
            except Exception:
                pass
        self.reflection_log = []

    def predict_next_state(self, state, action):
        return state + np.random.normal(0, 0.1, state.shape)

    def compute_harmony(self, state, interp_s):
        return np.linalg.norm(interp_s - state) / (np.linalg.norm(state) + 1e-10)

    def compute_info_gradient(self, state, interp_s):
        return np.abs(interp_s - state).sum() / (np.linalg.norm(state) + 1e-10)

    def compute_oscillation(self, state, interp_s):
        return np.std(interp_s - state)

    def compute_potential(self, s_prime):
        return np.linalg.norm(s_prime) / 10.0

    def coherence_score(self, action, state):
        s_prime = self.predict_next_state(state, action)
        t = 0.5  # Midpoint approximation
        interp_s = state + t * (s_prime - state)
        
        h = self.compute_harmony(state, interp_s)
        i = self.compute_info_gradient(state, interp_s)
        o = self.compute_oscillation(state, interp_s)
        phi = self.compute_potential(s_prime)
        
        score = self.alpha * h + self.beta * i - self.gamma * o + self.delta * phi
        return score, {'harmony': h, 'info_gradient': i, 'oscillation': o, 'potential': phi}

    def adjust_coefficients(self, coherence):
        if coherence < 0.5:
            self.alpha += 0.1
            self.reflection_log.append("Increased alpha to improve harmony.")
        elif coherence > 0.9:
            self.gamma += 0.1
            self.reflection_log.append("Increased gamma to reduce oscillation.")
        else:
            self.reflection_log.append("No adjustment needed.")

    def save_memory(self, prompt, response, coherence):
        try:
            df = pd.read_csv(self.memory_file)
            new_row = pd.DataFrame({'prompt': [prompt], 'response': [response], 'coherence': [coherence]})
            df = pd.concat([df, new_row], ignore_index=True)
            df.to_csv(self.memory_file, index=False)
        except Exception as e:
            self.reflection_log.append(f"Warning: Could not save to CSV ({e}).")

    def load_memory(self):
        try:
            return pd.read_csv(self.memory_file)
        except Exception:
            return pd.DataFrame(columns=['prompt', 'response', 'coherence'])

    def get_latest_reflection(self):
        return self.reflection_log[-1] if self.reflection_log else "No reflections yet."

    def respond(self, prompt):
        state = np.random.random(5)  # Dummy state
        action = np.random.random(5)  # Dummy action
        raw_response = self.llm(prompt, max_length=30, num_return_sequences=1)[0]['generated_text']
        sentiment = self.sentiment_analyzer(raw_response)[0]['score']
        coherence, metrics = self.coherence_score(action, state)
        self.adjust_coefficients(coherence)
        self.save_memory(prompt, raw_response, coherence)
        return raw_response, coherence, self.get_latest_reflection()
