import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from scipy import sparse
from scipy.io import arff
from keras.layers import Dense, Input, Dropout, concatenate
from keras.models import Model
import tensorflow as tf
from Method_change.Use_randomforest_reduce_dimension import reduce_dimension
from sklearn.decomposition import PCA


# from .arff get the data
data1, meta = arff.loadarff(open(r'D:\LAT_prediction\dataset\BioMedicalDataSet780s-actual-features.arff',
                                 encoding='utf-8'))
df = pd.DataFrame(data1)
dm = pd.DataFrame(meta)
print(df.shape)
print(list(df))

# get the labels and the attributes
labels = np.array(df.iloc[:, :85].values)
data = np.array(df.iloc[:, 85:].values)
labels1 = np.zeros((780, 85))

# transformer the str into int
for i in range(len(labels)):
    for j in range(len(labels[i])):
        labels1[i][j] = int(labels[i][j])

print(labels1.shape)
# print(labels1)
print(data.shape)

# ls: the shuffle of dataset, according to the crossvalidation class in Meka tool, get from ShuffleData.java
ls = [127, 540, 219, 400, 180, 270, 254, 693, 116, 438, 188, 513, 662, 351, 676, 566, 511, 491, 767, 609, 736, 308,
      349, 361, 326, 133, 176, 269, 607, 146, 666, 553, 779, 659, 150, 612, 442, 139, 768, 333, 309, 545, 406, 38,
      200, 430, 177, 528, 2, 339, 717, 484, 527, 660, 674, 640, 691, 103, 58, 277, 186, 396, 119, 5, 342, 30, 114,
      449, 387, 128, 728, 539, 774, 209, 90, 713, 704, 482, 476, 155, 162, 347, 647, 259, 148, 161, 32, 272, 310,
      450, 604, 625, 376, 617, 149, 568, 732, 718, 108, 293, 355, 585, 592, 372, 723, 159, 138, 357, 441, 187, 373,
      208, 411, 228, 181, 488, 273, 418, 753, 368, 532, 165, 99, 641, 285, 752, 531, 83, 469, 26, 233, 98, 485, 670,
      305, 53, 260, 615, 496, 199, 24, 445, 279, 64, 334, 156, 692, 140, 472, 516, 67, 677, 287, 275, 389, 475, 301,
      437, 497, 405, 317, 679, 778, 296, 374, 473, 120, 215, 306, 569, 167, 350, 763, 591, 561, 197, 244, 734, 560,
      299, 226, 737, 495, 409, 510, 689, 453, 613, 144, 217, 587, 773, 524, 512, 145, 504, 115, 122, 707, 257, 131,
      514, 431, 673, 262, 89, 739, 25, 79, 198, 567, 192, 129, 118, 327, 584, 724, 614, 460, 124, 135, 249, 688, 241,
      661, 91, 447, 303, 256, 563, 7, 669, 772, 113, 444, 125, 738, 705, 635, 121, 565, 628, 152, 107, 59, 322, 633,
      356, 202, 313, 486, 243, 599, 508, 34, 271, 498, 648, 41, 46, 466, 106, 525, 541, 395, 428, 632, 207, 84, 288,
      92, 134, 462, 652, 250, 680, 675, 3, 14, 29, 459, 573, 383, 94, 264, 100, 672, 164, 402, 761, 751, 332, 746, 182,
      440, 492, 694, 436, 382, 762, 172, 352, 631, 60, 683, 230, 696, 39, 630, 386, 353, 653, 318, 523, 314, 777, 49,
      6, 290, 726, 166, 620, 439, 377, 427, 194, 690, 667, 320, 130, 627, 11, 154, 68, 56, 490, 267, 255, 65, 426,
      721, 543, 744, 644, 731, 153, 85, 331, 239, 776, 102, 81, 371, 611, 638, 210, 18, 425, 419, 281, 348, 634,
      759, 765, 570, 461, 338, 754, 521, 709, 252, 642, 434, 710, 394, 423, 20, 289, 63, 743, 344, 758, 549, 312, 740,
      175, 454, 390, 479, 381, 142, 481, 229, 311, 636, 435, 464, 416, 624, 220, 76, 598, 218, 575, 54, 248, 536, 72,
      9, 388, 650, 170, 463, 294, 316, 346, 132, 96, 483, 766, 557, 78, 582, 722, 234, 658, 494, 22, 687, 412, 730,
      126, 282, 572, 246, 21, 757, 169, 205, 685, 66, 671, 602, 291, 367, 391, 101, 307, 448, 452, 1, 74, 618, 506,
      529, 242, 593, 123, 247, 714, 151, 595, 589, 37, 579, 586, 733, 489, 258, 80, 45, 185, 201, 468, 236, 522,
      455, 224, 238, 415, 530, 577, 656, 505, 547, 764, 771, 562, 538, 678, 55, 362, 284, 451, 725, 173, 745, 136,
      424, 622, 686, 283, 706, 232, 157, 477, 554, 23, 235, 325, 649, 596, 111, 203, 646, 471, 86, 370, 748, 87, 664,
      112, 27, 626, 263, 519, 668, 50, 364, 214, 398, 643, 749, 70, 298, 663, 407, 702, 401, 526, 330, 487, 715, 225,
      770, 147, 71, 727, 184, 0, 158, 190, 75, 82, 537, 564, 137, 43, 552, 433, 657, 295, 266, 15, 292, 760, 456, 542,
      69, 474, 336, 141, 274, 268, 315, 174, 109, 403, 206, 535, 533, 654, 712, 52, 699, 605, 729, 600, 48, 42, 193,
      212, 697, 583, 73, 12, 323, 360, 62, 77, 637, 544, 580, 684, 354, 493, 392, 574, 610, 328, 304, 581, 708, 601,
      95, 515, 413, 414, 551, 35, 240, 576, 379, 40, 603, 410, 44, 253, 211, 286, 204, 163, 681, 588, 223, 337, 105,
      61, 578, 237, 51, 178, 384, 629, 550, 695, 594, 369, 608, 28, 480, 300, 302, 88, 324, 458, 19, 742, 698, 231,
      507, 222, 571, 735, 597, 682, 4, 10, 443, 741, 623, 408, 478, 365, 517, 179, 590, 341, 359, 616, 420, 221, 397,
      33, 363, 378, 534, 196, 399, 183, 747, 189, 385, 501, 417, 393, 251, 465, 117, 719, 651, 31, 546, 665, 645,
      160, 171, 329, 655, 168, 621, 278, 421, 97, 769, 276, 429, 335, 619, 775, 261, 711, 470, 716, 366, 555, 518,
      104, 750, 559, 380, 340, 16, 606, 36, 558, 321, 701, 457, 280, 639, 500, 404, 213, 502, 446, 110, 265, 297,
      756, 345, 755, 520, 13, 195, 548, 245, 358, 216, 703, 47, 227, 343, 432, 509, 499, 93, 17, 191, 57, 8, 319,
      700, 143, 503, 467, 422, 375, 556, 720]


def draw_picture(new, train_dataset):
    # draw the picture of pca
    red_x, red_y = [], []
    blue_x, blue_y = [], []
    green_x, green_y = [], []

    for i in range(len(train_dataset)):
        if list(new[i]).index(1) == 34:
            red_x.append(train_dataset[i][0])
            red_y.append(train_dataset[i][1])
        elif list(new[i]).index(1) == 1:
            blue_x.append(train_dataset[i][0])
            blue_y.append(train_dataset[i][1])
        elif list(new[i]).index(1) == 26:
            green_x.append(train_dataset[i][0])
            green_y.append(train_dataset[i][1])

    plt.xlabel('component1(PCA)')
    plt.ylabel('component2(PCA)')
    plt.scatter(red_x, red_y, c='r', marker='x', label='label1')
    plt.scatter(blue_x, blue_y, c='b', marker='D', label='label2')
    plt.scatter(green_x, green_y, c='g', marker='.', label='label3')
    plt.legend()
    plt.savefig('pca.png')


def run(data, label, folds, estimators, epochs, num_features):
    data = data[ls, :]
    label = label[ls, :]
    number_of_instance = int(len(data) / folds)

    scores = []
    np.random.seed(1)
    for num in range(folds):
        # get the train data, test data
        start = num * number_of_instance
        end = (num + 1) * number_of_instance

        # print('start: ', start)
        # print('end: ', end)

        X_test = data[start: end, :]
        X_train = np.vstack((data[: start, :], data[end:, :]))
        # print("X_train: ", X_train.shape)
        y_test = label[start: end, :]

        y_test1 = sparse.lil_matrix((78, 85))
        y_test1[:, :] = y_test

        y_train = np.vstack((label[:start, :], label[end:, :]))

        feature_imp, feature_add, features_importances = reduce_dimension(X_train, y_train, num_features, estimators)
        features_importances = np.expand_dims(features_importances, 0).repeat(780, axis=0)
        total_label, new_label = get_processed_data(y_train)
        num_label_set = len(total_label)

        X_train1 = X_train[:, feature_imp]
        X_train1 *= (1 + 2 * features_importances[:702, :num_features])
        X_train2 = X_train[:, feature_add[:2400 - num_features]]
        X_train2 *= (1 + 1 * features_importances[:702, num_features: 2400])

        X_test1 = X_test[:, feature_imp]
        X_test1 *= (1 + 2 * features_importances[702:, :num_features])
        X_test2 = X_test[:, feature_add[:2400 - num_features]]
        X_test2 *= (1 + 1 * features_importances[702:, num_features: 2400])

        X_train2 = PCA(n_components=50).fit_transform(X_train2)
        draw_picture(new_label, X_train2)
        X_test2 = PCA(n_components=50).fit_transform(X_test2)

        input1 = Input(shape=(num_features,))
        input2 = Input(shape=(50,))
        Dropout(0.5)(input1)
        Dropout(0.5)(input2)

        l1 = Dense(512, activation='relu')(input1)
        l2 = Dense(32, activation='relu')(input2)

        hidden2 = concatenate([l2, l1], axis=-1)

        output = Dense(num_label_set, activation='softmax')(hidden2)

        model = Model([input1, input2], output)
        model.compile(loss='categorical_crossentropy', optimizer=tf.train.AdamOptimizer(),
                      metrics=['categorical_accuracy'])

        model.fit([X_train1, X_train2], new_label, epochs=epochs, batch_size=20)
        prediction1 = model.predict([X_test1, X_test2])

        # reconstruct the data
        predictions = []

        for i in prediction1:
            j = np.argmax(i)
            predictions.append(total_label[int(j)])

        y_pred = sparse.lil_matrix((78, 85))
        y_pred[:, :] = predictions

        score = f1_score(y_pred, y_test, average='micro')
        scores.append(score)
        print(score)

    print(sum(scores) / folds)
    return scores, sum(scores) / folds


# acoording to the Meka, setup the same parameter


def get_processed_data(label):
    # label (ndarray): use the label power set to change the form of labels
    total_label = []
    number = []

    for instance in label:
        instance = list(instance)
        if instance not in total_label:
            total_label.append(instance)
            number.append(1)
        else:
            pos = total_label.index(instance)
            number[pos] += 1

    for i in range(5):
        pos1 = number.index(max(number))
        print('pos', pos1)
        number[pos1] = 0

    # print('total_label', total_label)
    length = len(total_label)
    len1 = len(label)

    new_label = np.zeros((len1, length), dtype=int)

    for i in range(len1):
        if list(label[i]) in total_label:
            pos = total_label.index(list(label[i]))
            new_label[i][pos] = 1

    # print(new_label)

    return total_label, new_label


results = []
s = []

for i in range(1):
    ls1, result = run(data, labels1, 10, 101, 10, 1500)
    s.append(ls1)
    results.append(result)

# print(s)
# print(results)
# print(sum(results))
