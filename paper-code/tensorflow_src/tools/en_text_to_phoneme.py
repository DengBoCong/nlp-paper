def text_to_sequence(texts, max_len: int):
    """
    专用于phoneme的text转序列的方法
    :param texts: 文本序列列表
    :param max_len: 文本序列最大长度
    :return: 转换后的id序列
    """
    dict_set, _ = get_phoneme_dict_symbols()

    sequences = []
    for text in texts:
        sequence = []
        while len(text):
            # 判断有没有由花括号的音素，没有就直接按照字典转换
            m = re.compile(r'(.*?)\{(.+?)\}(.*)').match(text)
            if not m:
                sequence += [dict_set[s] for s in _clean_text(text)
                             if s in dict_set and s is not 'unk' and s is not '~']
                break
            sequence += [dict_set[s] for s in _clean_text(m.group(1))
                         if s in dict_set and s is not 'unk' and s is not '~']
            sequence += [dict_set[s] for s in ['@' + s for s in m.group(2).split()]
                         if s in dict_set and s is not 'unk' and s is not '~']
            text = m.group(3)
        sequence.append(dict_set['~'])
        sequences.append(sequence)
    sequences = tf.keras.preprocessing.sequence.pad_sequences(
        sequences, maxlen=max_len, padding="post")
    return sequences


def text_to_phonemes_converter(text: str, cmu_dict_path: str):
    """
    将句子按照CMU音素字典进行分词切分
    :param text: 单个句子文本
    :param cmu_dict_path: cmu音素字典路径
    :return: 按照音素分词好的数组
    """
    _, symbols_set = get_phoneme_dict_symbols()

    alt_re = re.compile(r'\([0-9]+\)')
    cmu_dict = {}
    text = _clean_text(text)
    text = re.sub(r"([?.!,])", r" \1", text)

    # 文件是从官网下载的，所以文件编码格式要用latin-1
    with open(cmu_dict_path, 'r', encoding='latin-1') as cmu_file:
        for line in cmu_file:
            if len(line) and (line[0] >= "A" and line[0] <= "Z" or line[0] == "'"):
                parts = line.split('  ')
                word = re.sub(alt_re, '', parts[0])

                # 这里要将非cmu音素的干扰排除
                pronunciation = " "
                temps = parts[1].strip().split(' ')
                for temp in temps:
                    if temp not in symbols_set:
                        pronunciation = None
                        break
                if pronunciation:
                    pronunciation = ' '.join(temps)
                    if word in cmu_dict:
                        cmu_dict[word].append(pronunciation)
                    else:
                        cmu_dict[word] = [pronunciation]

    cmu_result = []
    for word in text.split(' '):
        # 因为同一个单词，它的发音音素可能不一样，所以存在多个
        # 音素分词，我这里就单纯的取第一个，后面再改进和优化
        cmu_word = cmu_dict.get(word.upper(), [word])[0]
        if cmu_word != word:
            cmu_result.append("{" + cmu_word + "}")
        else:
            cmu_result.append(cmu_word)

    return " ".join(cmu_result)

def _clean_text(text: str):
    """
    用于对句子进行整理，将美元、英镑、数字、小数点、序
    数词等等转化为单词，同时对部分缩写进行扩展
    :param text: 单个句子文本
    :return: 处理好的文本序列
    """
    text = unidecode(text)
    text = text.lower()
    text = _clean_number(text=text)
    text = _abbreviations_to_word(text=text)
    text = re.sub(r"\s+", " ", text)

    return text


def get_phoneme_dict_symbols(unknown: str = "<unk>", eos: str = "~"):
    """
    用于创建音素文件，方便在pre_treat中使用
    :param unknown: 未登录词
    :param eos: 结尾词
    :return: 字典和39个原始音素和字符的集合
    """
    symbols = [
        'AA', 'AA0', 'AA1', 'AA2', 'AE', 'AE0', 'AE1', 'AE2', 'AH', 'AH0', 'AH1', 'AH2',
        'AO', 'AO0', 'AO1', 'AO2', 'AW', 'AW0', 'AW1', 'AW2', 'AY', 'AY0', 'AY1', 'AY2',
        'B', 'CH', 'D', 'DH', 'EH', 'EH0', 'EH1', 'EH2', 'ER', 'ER0', 'ER1', 'ER2', 'EY',
        'EY0', 'EY1', 'EY2', 'F', 'G', 'HH', 'IH', 'IH0', 'IH1', 'IH2', 'IY', 'IY0', 'IY1',
        'IY2', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OW0', 'OW1', 'OW2', 'OY', 'OY0',
        'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UH0', 'UH1', 'UH2', 'UW',
        'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH'
    ]

    chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!\'(),-.:;? '
    phonemes = ['@' + s for s in symbols]
    symbols_list = [unknown, eos] + list(chars) + phonemes

    dict_set = {s: i for i, s in enumerate(symbols_list)}

    return dict_set, set(symbols)


def _clean_number(text: str):
    """
    对句子中的数字相关进行统一单词转换
    :param text: 单个句子文本
    :return: 转换后的句子文本
    """
    comma_number_re = re.compile(r"([0-9][0-9\,]+[0-9])")
    decimal_number_re = re.compile(r"([0-9]+\.[0-9]+)")
    pounds_re = re.compile(r"£([0-9\,]*[0-9]+)")
    dollars_re = re.compile(r"\$([0-9\.\,]*[0-9]+)")
    ordinal_re = re.compile(r"[0-9]+(st|nd|rd|th)")
    number_re = re.compile(r"[0-9]+")

    text = re.sub(comma_number_re, lambda m: m.group(1).replace(',', ''), text)
    text = re.sub(pounds_re, r"\1 pounds", text)
    text = re.sub(dollars_re, _dollars_to_word, text)
    text = re.sub(decimal_number_re, lambda m: m.group(
        1).replace('.', ' point '), text)
    text = re.sub(ordinal_re, lambda m: inflect.engine(
    ).number_to_words(m.group(0)), text)
    text = re.sub(number_re, _number_to_word, text)

    return text


def _number_to_word(number_re: str):
    """
    将数字转为单词
    :param number_re: 数字匹配式
    :return:
    """
    num = int(number_re.group(0))
    tool = inflect.engine()

    if 1000 < num < 3000:
        if num == 2000:
            return "two thousand"
        elif 2000 < num < 2010:
            return "two thousand " + tool.number_to_words(num % 100)
        elif num % 100 == 0:
            return tool.number_to_words(num // 100) + " hundred"
        else:
            return tool.number_to_words(num, andword="", zero='oh', group=2).replace(", ", " ")
    else:
        return tool.number_to_words(num, andword="")


def _dollars_to_word(dollars_re: str):
    """
    将美元转为单词
    :param dollars_re: 美元匹配式
    :return:
    """
    match = dollars_re.group(1)
    parts = match.split('.')
    if len(parts) > 2:
        # 注意力，不符合格式的要直接返回
        return match + ' dollars'
    dollars = int(parts[0]) if parts[0] else 0
    cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    if dollars and cents:
        dollar_unit = 'dollar' if dollars == 1 else 'dollars'
        cent_unit = 'cent' if cents == 1 else 'cents'
        return '%s %s, %s %s' % (dollars, dollar_unit, cents, cent_unit)
    elif dollars:
        dollar_unit = 'dollar' if dollars == 1 else 'dollars'
        return '%s %s' % (dollars, dollar_unit)
    elif cents:
        cent_unit = 'cent' if cents == 1 else 'cents'
        return '%s %s' % (cents, cent_unit)
    else:
        return 'zero dollars'


def _abbreviations_to_word(text: str):
    """
    对句子中的压缩次进行扩展成单词
    :param text: 单个句子文本
    :return: 转换后的句子文本
    """
    abbreviations = [
        (re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
            ('mrs', 'misess'),
            ('mr', 'mister'),
            ('dr', 'doctor'),
            ('st', 'saint'),
            ('co', 'company'),
            ('jr', 'junior'),
            ('maj', 'major'),
            ('gen', 'general'),
            ('drs', 'doctors'),
            ('rev', 'reverend'),
            ('lt', 'lieutenant'),
            ('hon', 'honorable'),
            ('sgt', 'sergeant'),
            ('capt', 'captain'),
            ('esq', 'esquire'),
            ('ltd', 'limited'),
            ('col', 'colonel'),
            ('ft', 'fort')
        ]
    ]

    for regex, replacement in abbreviations:
        text = re.sub(regex, replacement, text)

    return text
