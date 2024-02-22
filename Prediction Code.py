def predict_ddos(pktcount, bytecount, protocol_icmp, protocol_udp):
# Load the trained model
clf = joblib.load('/content/drive/MyDrive/DDoS Detection/lime_model.pkl')
# Create a feature array with the selected features
X = np.array([[pktcount, bytecount, protocol_icmp, protocol_udp]])
# Predict label
y_pred = clf.predict(X)
return y_pred[0]
# Input from user
pktcount = float(input("pktcount: "))
bytecount = float(input("bytecount: "))
protocol_icmp = float(input("Protocol_TCP: "))
protocol_udp = float(input("Protocol_UDP: "))
# Call predict function
y_pred = predict_ddos(pktcount, bytecount, protocol_icmp, protocol_udp)
print()
if y_pred == 0:
print("Traffic is benign (0).")
else:
print("Traffic is malicious (1).")