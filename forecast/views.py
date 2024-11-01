from django.shortcuts import render

# Create your views here.
import feedparser
import pandas as pd
from transformers import pipeline
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.alert import Alert
from selenium.webdriver.chrome.options import Options
import pandas as pd
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import yfinance as yf



#  Web Scraping 
def getSentiments():
    url = "https://www.ft.com/world?format=rss"
    feed = feedparser.parse(url)
    articles = []
    for entry in feed.entries:
        articles.append({
            "Title": entry.title,
            "Link": entry.link,
            "Published Date": entry.published
        })
    df = pd.DataFrame(articles)
    df.columns = ['Title', 'Link', 'Date']
    
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    sentiment_pipeline = pipeline("sentiment-analysis")
    df['Sentiment'] = df['Title'].apply(lambda text: sentiment_pipeline(text)[0]['score'])
    df[['Date','Sentiment']].groupby('Date').mean()
    return df[['Date','Sentiment']].groupby('Date').mean()


def getUS30datausing_Selenium():
    driver = webdriver.Edge(r"C:\Users\tv078\chromedriver_win32\msedgedriver.exe")
    driver.get("https://in.tradingview.com/symbols/BLACKBULL-US30/")
    dictonary = {}

    hidden_content = WebDriverWait(driver, 10).until(
        EC.visibility_of_element_located((By.XPATH, '//*[@id="js-category-content"]/div[2]/div/section/div[3]/div[2]/div/div/div[1]/div[2]/div[1]/div'))
    )
    dictonary['Volume'] = hidden_content.text
    hidden_content = WebDriverWait(driver, 10).until(
        EC.visibility_of_element_located((By.XPATH, '//*[@id="js-category-content"]/div[2]/div/section/div[3]/div[2]/div/div/div[2]/div[2]/div[1]/div'))
    )
    dictonary['Close'] = hidden_content.text

    hidden_content = WebDriverWait(driver, 10).until(
        EC.visibility_of_element_located((By.XPATH, '//*[@id="js-category-content"]/div[2]/div/section/div[3]/div[2]/div/div/div[3]/div[2]/div[1]/div'))
    )
    dictonary['Open'] = hidden_content.text

    hidden_content = WebDriverWait(driver, 10).until(
        EC.visibility_of_element_located((By.XPATH, '//*[@id="js-category-content"]/div[2]/div/section/div[3]/div[2]/div/div/div[4]/div[2]/div[1]/div/div/span[1]'))
    )
    dictonary['Day_range_from'] = hidden_content.text

    hidden_content = WebDriverWait(driver, 10).until(
        EC.visibility_of_element_located((By.XPATH, '//*[@id="js-category-content"]/div[2]/div/section/div[3]/div[2]/div/div/div[4]/div[2]/div[1]/div/div/span[2]/span[1]'))
    )
    dictonary['Day_range_to'] = hidden_content.text
    return dictonary

def get_Historical_US30_data():
    
    ticker = "^DJI"  # US30 
    data = yf.Ticker(ticker)
    historical_data = data.history(period="5d")
    historical_data.index=historical_data.index.date
    return historical_data
    
    
def dataIntigration():
    us30_data= get_Historical_US30_data()
    news_data = getSentiments()
    merged_data = pd.merge(us30_data, news_data, on="Date", how="left").fillna(0)
    return merged_data



def proprocess_Data():
    merged_data = dataIntigration()
    price_data = scaler.fit_transform(merged_data[['Open', 'Close']])
    
    X_prices = price_data
    X_sentiments = merged_data['Sentiment'].values.reshape(-1, 1)  # Add sentiment as an additional feature

    X = np.concatenate([X_prices, X_sentiments], axis=1)
    y = price_data[:, -1] 
    
    sequence_length = 2
    X_sequences = np.array([X[i:i + sequence_length] for i in range(len(X) - sequence_length)])
    y_sequences = np.array([y[i + sequence_length] for i in range(len(y) - sequence_length)])
    
    split_idx = int(len(X_sequences) * 0.8)
    return X_sequences[:split_idx], y_sequences[:split_idx], X_sequences[split_idx:], y_sequences[split_idx:], scaler
        
        



class TransformerTimeSeries(nn.Module):
    def __init__(self, feature_size=5, num_layers=2, num_heads=2, seq_length=2, hidden_dim=64):
        super(TransformerTimeSeries, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=num_heads, dim_feedforward=hidden_dim)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(seq_length * feature_size, 1) 

    def forward(self, x):
        x = self.transformer(x)
        x = x.view(x.size(0), -1)  
        return self.fc(x)



def forecast_view(request):
    merged_data = dataIntigration()
    X_train, y_train, X_test, y_test, scaler = proprocess_Data()

    # Load model
    model = TransformerTimeSeries()
    model.eval()  # Set model to evaluation mode

    # Training and test predictions (in-sample)
    X_tensor_train = torch.tensor(X_train, dtype=torch.float32)
    train_predictions = model(X_tensor_train).squeeze().detach().numpy()
    actual_train = scaler.inverse_transform(np.c_[np.zeros((train_predictions.shape[0], 3)), y_train])
    predicted_train = scaler.inverse_transform(np.c_[np.zeros((train_predictions.shape[0], 3)), train_predictions])

    X_tensor_test = torch.tensor(X_test, dtype=torch.float32)
    test_predictions = model(X_tensor_test).squeeze().detach().numpy()
    actual_test = scaler.inverse_transform(np.c_[np.zeros((test_predictions.shape[0], 3)), y_test])
    predicted_test = scaler.inverse_transform(np.c_[np.zeros((test_predictions.shape[0], 3)), test_predictions])

    # Forecast
    forecast_sequences = X_test[-30:]  # Using last 30 sequences
    forecasted_prices = []
    for _ in range(30):
        forecast_input = torch.tensor(forecast_sequences, dtype=torch.float32)
        forecast_price = model(forecast_input).item()
        forecasted_prices.append(forecast_price)
        forecast_sequences = np.roll(forecast_sequences, -1, axis=0)
        forecast_sequences[-1, -1] = forecast_price

    forecasted_prices = scaler.inverse_transform(np.c_[np.zeros((len(forecasted_prices), 3)), forecasted_prices])

    # Generate plot
    plt.figure(figsize=(12, 6))
    plt.plot(actual_train[:, -1], label="Actual (Train)")
    plt.plot(predicted_train[:, -1], label="Predicted (Train)")
    plt.plot(range(len(actual_train), len(actual_train) + len(actual_test)), actual_test[:, -1], label="Actual (Test)")
    plt.plot(range(len(actual_train), len(actual_train) + len(actual_test)), predicted_test[:, -1], label="Predicted (Test)")
    plt.plot(range(len(actual_train) + len(actual_test), len(actual_train) + len(actual_test) + len(forecasted_prices)), forecasted_prices[:, -1], label="Forecasted", linestyle="--")
    plt.legend()

    # Save plot to display in Django template
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    buf.close()

    return render(request, 'forecast/forecast.html', {'image_base64': image_base64})
