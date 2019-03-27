from PWSegmentation import PWSegmentation
import cv2
import os

seg = PWSegmentation("annotate/Crope",
             ["annotate/class_dors_2",
              "annotate/class_anna",
              "annotate/class_back",
              "annotate/class_body",
              "annotate/class_dors_1",
              "annotate/class_caud",
              "annotate/class_pect",
              "annotate/class_pelvi"],
             ["dors_2",
              "anna",
              "back",
              "body",
              "dors_1",
              "caud",
              "pect",
              "pelvi"],
             1/4
             )


seg.concat_data()
seg.train(ntrees=40)
bin_img_predict = seg.predict(cv2.imread("pred/img/BAR12 (6).jpg"))

img_name = os.listdir("pred/img/")

for i in range(len(img_name)):
    print("pred/img/" + img_name[i])
    mask_list, summary = seg.predict(cv2.imread("pred/img/" + img_name[i]), display=False)
    cv2.imwrite("pred/class_dors_2/" + img_name[i] + ".jpg", mask_list[0])
    cv2.imwrite("pred/class_anna/" + img_name[i] + ".jpg", mask_list[1])
    cv2.imwrite("pred/class_back/" + img_name[i] + ".jpg", mask_list[2])
    cv2.imwrite("pred/class_body/" + img_name[i] + ".jpg", mask_list[3])
    cv2.imwrite("pred/class_dors_1/" + img_name[i] + ".jpg", mask_list[4])
    cv2.imwrite("pred/class_caud/" + img_name[i] + ".jpg", mask_list[5])
    cv2.imwrite("pred/class_pect/" + img_name[i] + ".jpg", mask_list[6])
    cv2.imwrite("pred/class_pelvi/" + img_name[i] + ".jpg", mask_list[7])
    cv2.imwrite("pred/summary/" + img_name[i] + ".jpg", summary)


