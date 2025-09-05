try:
    import yfinance as yf
    import matplotlib.pyplot as plt

    # Fetch monthly QQQ stock price data
    print("Fetching QQQ data...")
    qqq_data = yf.download('QQQ', period='max', interval='1mo')

    if qqq_data.empty:
        print("No data was retrieved. Check your internet connection or the ticker symbol.")
    else:
        # Plot the monthly closing prices
        plt.figure(figsize=(12, 6))
        plt.plot(qqq_data.index, qqq_data['Close'], label='Monthly Close Price', color='blue')
        plt.title('QQQ Monthly Closing Prices')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.grid(True)

        # Save the graph as 'qqq_monthly.png'
        plt.savefig('qqq_monthly.png')
        plt.close()

        print("Graph saved as 'qqq_monthly.png'")

except ImportError as e:
    print(f"Missing dependency: {e}. Please install the required package using:")
    print("pip install yfinance matplotlib")
except Exception as e:
    print(f"An error occurred: {e}")