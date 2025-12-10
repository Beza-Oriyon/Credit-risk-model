# Credit-risk-model
Credit scoring model by using ecommerce data provided by ecommerce plateform for Beti bank
## Credit-Scoring-Business-Understanding
### 1.
### 2. Since Beti bank uses alternative ecommerce data instead of default label we have to create substitute variable that acts as stand in fort rue risk which is Proxy value.** We must create a **proxy variable** to define a "bad" customer because the model needs a target variable to learn the distinction between good and bad risk. A common proxy is **90+ days past due (DPD)** on any loan within an observation window.
**The business risks caused by using the proxy value is measurment error which is false positive(TypeI error) and false negative(TypeII error)
###3.**Simple Logistic Model- High interpretability , easy to explain, mostly used(prefered) in regulatory banking, shorter development cycle, **Accepting slightly lower performance** for maximum **transparency** and **regulatory compliance.
**Complex(Gradient Boosting): low interprtability, hard to explain, used for high performance , used when interpretability is not priority,Longer development cycle, **Achieving maximum performance** at the cost of **transparency** and increased **regulatory/audit overhead.**
