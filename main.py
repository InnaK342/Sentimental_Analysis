from customtkinter import *
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import joblib
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk


class App:

    def __init__(self):
        self.window = CTk()
        self.window.title('Sentiment Analysis App')
        self.window.geometry('600x650')

        self.text = CTkLabel(self.window, text='Enter the text of the post:', font=('Apple SD Gothic Neo', 18))
        self.text.place(x=190, y=60)

        self.textbox = CTkTextbox(self.window, width=400, height=100, font=('Apple SD Gothic Neo', 12))
        self.textbox.place(x=90, y=100)

        self.button = CTkButton(self.window, text='Result', font=('Apple SD Gothic Neo', 15), command=self.process_text)
        self.button.place(x=215, y=210)

        self.result = CTkLabel(self.window, text='', font=('Apple SD Gothic Neo', 15))
        self.result.place(x=190, y=247)

        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.window.mainloop()


    def process_text(self):
        vectorizer = joblib.load('models/vectorizer_2.pkl')
        scaler = joblib.load('models/standard_scaler_2.pkl')
        model = joblib.load('models/model_2.pkl')

        text = self.textbox.get('0.0', 'end')

        review = re.sub('[^a-zA-Z]', ' ', text)
        review = review.lower().split()
        # ps = PorterStemmer()
        # review = [ps.stem(word) for word in review if word not in set(stopwords.words('english'))]
        review = ' '.join(review)
        x = vectorizer.transform([review]).toarray()
        x = scaler.transform(x)

        result = model.predict(x)[0]
        percentages = model.predict_proba(x)
        self.result.configure(text=f'Result: {"negative" if result == -1 else "positive" if result == 1 else "neutral"}')
        self.draw_graph(percentages)

    def draw_graph(self, percentages):
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.set_facecolor('#E5E5E5')
        ax.bar(['negative', 'neutral', 'positive'], percentages[0], color='#5EAFE5')
        plt.title('Graphical Representation', fontname='Apple SD Gothic Neo', fontsize=10)
        plt.ylabel('Percentage', fontname='Apple SD Gothic Neo', fontsize=10)
        ax.grid()

        canvas = FigureCanvasTkAgg(fig, master=self.window)
        canvas.get_tk_widget().place(x=50, y=280)

    def on_closing(self):
        plt.close()
        self.window.destroy()



app = App()
