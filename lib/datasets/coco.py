COCO_PERSON_SKELETON = [[0,1],[1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8]]
    
COCO_KEYPOINTS = [
    'First_point',
    'Second_point',
    'Third_point',
    'Fourth_point',
    'Fifth_point',
    'Sixth_point',
    'Seventh_point',
    'Eighth_point',
    'Ninth_point'
]


HFLIP = {
    'First_point':'First_point',
    'Second_point':'Second_point',
    'Third_point':'Third_point',
    'Fourth_point':'Fourth_point',
    'Fifth_point':'Fifth_point',
    'Sixth_point':'Sixth_point',
    'Seventh_point':'Seventh_point',
    'Eighth_point':'Eighth_point',
    'Ninth_point':'Ninth_point',
}

COCO_PERSON_SIGMAS = [
    0.026,  # nose
    0.025,  # eyes
    0.025,  # eyes
    0.035,  # ears
    0.035,  # ears
    0.079,  # shoulders
    0.079,  # shoulders
    0.072,  # elbows
    0.072,  # elbows
    0.062,  # wrists
    0.062,  # wrists
    0.107,  # hips
    0.107,  # hips
    0.087,  # knees
    0.087,  # knees
    0.089,  # ankles
    0.089,  # ankles
]

def print_associations():
    for j1, j2 in COCO_PERSON_SKELETON:
        print(COCO_KEYPOINTS[j1], '-', COCO_KEYPOINTS[j2])


if __name__ == '__main__':
    print_associations()
    # draw_skeletons()
