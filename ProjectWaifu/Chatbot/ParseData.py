import re  # regex


def cornell_cleanup(sentence):
    # clean up html tags
    sentence = re.sub(r'<.*?>', '', sentence.lower())
    # clean up \n and \r
    return sentence.replace('\n', '').replace('\r', '')


def load_cornell(path_conversations, path_lines):
    movie_lines = {}
    lines_file = open(path_lines, 'r', encoding="iso-8859-1")
    for line in lines_file:
        line = line.split(" +++$+++ ")
        line_number = line[0]
        character = line[1]
        movie = line[2]
        sentence = line[-1]

        if movie not in movie_lines:
            movie_lines[movie] = {}

        movie_lines[movie][line_number] = (character, sentence)

    questions = []
    responses = []
    conversations_file = open(path_conversations, 'r', encoding="iso-8859-1")
    for line in conversations_file:
        line = line.split(" +++$+++ ")
        movie = line[2]
        line_numbers = []
        for num in line[3][1:-2].split(", "):
            line_numbers.append(num[1:-1])

        # Not used since the cornell data set already placed
        # the lines of the same character together
        #
        # lines = []
        #
        # tmp = []
        #
        # teacher = movie_lines[movie][line_numbers[0]][0]
        # # teacher is the one that speaks first
        # was_teacher = True
        #
        # for num in line_numbers:
        #
        #     line = movie_lines[movie][num]
        #     if line[0] == teacher:
        #         if not was_teacher:  # was the bot
        #             lines.append([True, tmp])  # append previous conversation and mark as "is bot"
        #             tmp = []
        #         tmp.append(cornell_cleanup(line[1]))
        #         was_teacher = True
        #     else:  # bot speaking
        #         if was_teacher:  # was teacher
        #             lines.append([False, tmp])  # append previous conversation and mark "is not bot"
        #             tmp = []
        #         tmp.append(cornell_cleanup(line[1]))
        #         was_teacher = False
        #
        # if len(tmp) > 0:
        #     lines.append([not was_teacher, tmp])  # append the last response (not b/c of the inverse)
        #
        # conversations.append(lines)

        for i in range(len(line_numbers) - 1):
            questions.append(cornell_cleanup(movie_lines[movie][line_numbers[i]][1]))
            responses.append(cornell_cleanup(movie_lines[movie][line_numbers[i + 1]][1]))

    return questions, responses


def load_twitter(path):
    lines_x = []
    lines_y = []

    lines = open(path, 'r', encoding='utf-8')

    is_x = True

    for line in lines:
        if is_x:
            lines_x.append(line.lower())
        else:
            lines_y.append(line.lower())
        is_x = not is_x

    return lines_x, lines_y


def split_sentence(sentence):
    # collect independent words
    result = re.findall(r"[\w]+|[.,!?;'\"]+", sentence)
    return result


def split_data(data):
    result = []
    for line in data:
        result.append(split_sentence(line))
    return result


def sentence_to_index(sentence, word_to_index, target=False):
    if not target:
        result = [word_to_index["<GO>"]]
        length = 1
    else:
        result = []
        length = 0
    unk = 0
    for word in sentence:
        length += 1
        if word in word_to_index:
            result.append(word_to_index[word])
        else:
            result.append(word_to_index["<UNK>"])
            unk += 1

    # max sequence length of 20
    if length < 20:
        result.append(word_to_index["<EOS>"])
        length += 1
        # EOS also used as padding
        result.extend([word_to_index["<EOS>"]] * (20 - length))
    else:
        # result = result[:19]
        # result.append(word_to_index["<EOS>"])
        # length = 19
        result = result[:20]
        length = 20

    return result, length, unk


def data_to_index(data_x, data_y, word_to_index):
    result_x = []
    result_y = []
    lengths_x = []
    lengths_y = []
    result_y_target = []
    index = 0

    while index < len(data_x):
        x, x_length, x_unk = sentence_to_index(data_x[index], word_to_index)
        y, y_length, y_unk = sentence_to_index(data_y[index], word_to_index)

        index += 1

        if x_unk > 1 or y_unk > 0:
            continue

        result_x.append(x)
        result_y.append(y)
        lengths_x.append(x_length)
        lengths_y.append(y_length)

        y_target = y[1:]
        y_target.append(word_to_index["<EOS>"])
        result_y_target.append(y_target)

    return result_x, result_y, lengths_x, lengths_y, result_y_target
