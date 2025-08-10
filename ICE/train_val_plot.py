import matplotlib.pyplot as plt

# 에폭별 Loss와 Accuracy 기록 (네가 앞에서 얻은 값들을 넣는 거야)
train_loss = [0.7347, 0.1281, 0.0766, 0.0506, 0.0288, 0.0547, 0.1418, 0.0936, 0.0525, 0.0588]
val_acc =    [91.15, 88.50, 93.81, 92.04, 94.25, 93.81, 81.42, 92.48, 91.15, 93.36]

epochs = range(1, len(train_loss) + 1)

# 서브플롯: Loss와 Accuracy
plt.figure(figsize=(12, 5))

# 📉 Loss 그래프
plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, marker='o', label='Train Loss')
plt.title('Train Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()

# 📈 Accuracy 그래프
plt.subplot(1, 2, 2)
plt.plot(epochs, val_acc, marker='o', color='orange', label='Validation Accuracy')
plt.title('Validation Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("C:/ICE/result.png")
plt.show()
