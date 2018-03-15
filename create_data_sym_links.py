import os
data_dir = 'data'
left_image_dir = 'image_2'
right_image_dir = 'image_3'

# train_sequences = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09',
#                    '11', '12', '13', '14', '15', '16', '17', '18', '19', '20']

train_sequences = ['05', '12']

test_sequences = ['10']

train_dir = os.path.join(data_dir, 'train')

test_dir = os.path.join(data_dir, 'test')

if not os.path.exists(train_dir):
    os.makedirs(train_dir)

if not os.path.exists(test_dir):
    os.makedirs(test_dir)

if not os.path.exists(os.path.join(train_dir, 'left')):
    os.makedirs(os.path.join(train_dir, 'left'))

if not os.path.exists(os.path.join(train_dir, 'right')):
    os.makedirs(os.path.join(train_dir, 'right'))

if not os.path.exists(os.path.join(test_dir, 'left')):
    os.makedirs(os.path.join(test_dir, 'left'))

if not os.path.exists(os.path.join(test_dir, 'right')):
    os.makedirs(os.path.join(test_dir, 'right'))

root_dir = os.getcwd()

for seq in train_sequences:
    l_dir = os.path.join(data_dir, 'dataset', 'sequences', seq, left_image_dir)
    r_dir = os.path.join(data_dir, 'dataset', 'sequences', seq, right_image_dir)
    for (_, _, left_filenames) in os.walk(l_dir):
        break
    for (_, _, right_filenames) in os.walk(r_dir):
        break

    for file in left_filenames:
        src_path = os.path.join(root_dir, l_dir, file)
        dst_path = os.path.join(train_dir, 'left', seq+file)
        os.symlink(src_path, dst_path)

    for file in right_filenames:
        src_path = os.path.join(root_dir, r_dir, file)
        dst_path = os.path.join(train_dir, 'right', seq+file)
        os.symlink(src_path, dst_path)

for seq in test_sequences:
    l_dir = os.path.join(data_dir, 'dataset', 'sequences', seq, left_image_dir)
    r_dir = os.path.join(data_dir, 'dataset', 'sequences', seq, right_image_dir)
    for (_, _, left_filenames) in os.walk(l_dir):
        break
    for (_, _, right_filenames) in os.walk(r_dir):
        break

    for file in left_filenames:
        src_path = os.path.join(root_dir, l_dir, file)
        dst_path = os.path.join(test_dir, 'left', seq+file)
        os.symlink(src_path, dst_path)

    for file in right_filenames:
        src_path = os.path.join(root_dir, r_dir, file)
        dst_path = os.path.join(test_dir, 'right', seq+file)
        os.symlink(src_path, dst_path)
