import tensorflow as tf

model = tf.keras.models.load_model(
    r'C:\Users\yeseo\PycharmProjects\Infostealer_Detection\stealer_service\model\stealer_model.keras'
)
print('정상! 모델 로드 성공')

print(model.summary())