#head of the data
--- 1. Head of the Data ---
         TransactionId         BatchId       AccountId       SubscriptionId  \
0  TransactionId_76871   BatchId_36123  AccountId_3957   SubscriptionId_887   
1  TransactionId_73770   BatchId_15642  AccountId_4841  SubscriptionId_3829   
2  TransactionId_26203   BatchId_53941  AccountId_4229   SubscriptionId_222   
3    TransactionId_380  BatchId_102363   AccountId_648  SubscriptionId_2185   
4  TransactionId_28195   BatchId_38780  AccountId_4841  SubscriptionId_3829   

        CustomerId CurrencyCode  CountryCode    ProviderId     ProductId  \
0  CustomerId_4406          UGX          256  ProviderId_6  ProductId_10   
1  CustomerId_4406          UGX          256  ProviderId_4   ProductId_6   
2  CustomerId_4683          UGX          256  ProviderId_6   ProductId_1   
3   CustomerId_988          UGX          256  ProviderId_1  ProductId_21   
4   CustomerId_988          UGX          256  ProviderId_4   ProductId_6   

      ProductCategory    ChannelId   Amount  Value  TransactionStartTime  \
0             airtime  ChannelId_3   1000.0   1000  2018-11-15T02:18:49Z   
1  financial_services  ChannelId_2    -20.0     20  2018-11-15T02:19:08Z   
2             airtime  ChannelId_3    500.0    500  2018-11-15T02:44:21Z   
3        utility_bill  ChannelId_3  20000.0  21800  2018-11-15T03:32:55Z   
4  financial_services  ChannelId_2   -644.0    644  2018-11-15T03:34:21Z   

   PricingStrategy  FraudResult  
0                2            0  
1                2            0  
...
dtypes: float64(1), int64(4), object(11)
memory usage: 11.7+ MB

Total number of unique customers: 3742
Output is truncated. View as a scrollable element or open in a text editor. Adjust cell output settings...