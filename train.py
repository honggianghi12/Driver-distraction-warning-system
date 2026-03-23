import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Tạo bộ dữ liệu với ImageDataGenerator
datagen = ImageDataGenerator(
    rescale=1./255, validation_split=0.2)  # Chia 80% train, 20% val

# Load dữ liệu huấn luyện
train_data = datagen.flow_from_directory(
    'processed_dataset/',
    target_size=(24, 24),  # Kích thước ảnh đầu vào
    color_mode='grayscale',
    batch_size=32,
    class_mode='binary',
    subset='training'
)

# Load dữ liệu validation
val_data = datagen.flow_from_directory(
    'processed_dataset/',
    target_size=(24, 24),
    color_mode='grayscale',
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# Xây dựng mô hình CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu',
           input_shape=(24, 24, 1)),  # 32 bộ lọc
    # Lớp giảm kích thước đi 1 nửa, số chiều sâu vẵn giữu nguyên
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),  # 64 bộ lọc
    MaxPooling2D(2, 2),
    Flatten(),  # Biến tensor 2D thành vector 1D
    Dense(128, activation='relu'),  # Fully connected layer
    Dropout(0.5),  # Tránh overfitting
    Dense(1, activation='sigmoid')  # 1 output: mắt mở (1) hoặc nhắm (0)
])

# Biên dịch mô hình
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])

# Huấn luyện mô hình
model.fit(train_data, validation_data=val_data, epochs=10)

# Lưu mô hình sau khi huấn luyện
model.save('eye_model.h5')

print("✔ Huấn luyện hoàn tất, mô hình đã được lưu vào eye_model.h5!")
