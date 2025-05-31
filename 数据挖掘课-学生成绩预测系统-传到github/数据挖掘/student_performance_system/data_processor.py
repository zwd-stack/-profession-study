#data_processor.py
class DataProcessor:
    def __init__(self):
        self.numeric_columns = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime',
                              'failures', 'famrel', 'freetime', 'goout', 'Dalc',
                              'Walc', 'health', 'absences', 'G1', 'G2', 'G3']
        self.categorical_columns = ['school', 'sex', 'address', 'famsize', 'Pstatus',
                                  'Mjob', 'Fjob', 'reason', 'guardian', 'schoolsup',
                                  'famsup', 'paid', 'activities', 'nursery', 'higher',
                                  'internet', 'romantic']
        self.target_column = 'G3'

    def load_data(self, filepath):
        """加载CSV数据"""
        data = {}
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                headers = f.readline().strip().split(';')
                for header in headers:
                    data[header] = []

                # 读取数据
                for line in f:
                    values = line.strip().split(';')
                    for i, value in enumerate(values):
                        data[headers[i]].append(value)

            return data, headers
        except Exception as e:
            raise Exception(f"数据加载错误: {str(e)}")

    def preprocess_data(self, data, standardize=True, handle_missing=True):
        """数据预处理"""
        processed_data = {}

        for col in data:
            processed_data[col] = data[col].copy()

        if handle_missing:
            processed_data = self._handle_missing_values(processed_data)


        processed_data = self._encode_categorical_variables(processed_data)

        if standardize:
            processed_data = self._standardize_numeric_features(processed_data)

        return processed_data

    def _handle_missing_values(self, data):
        """处理缺失值"""
        for col in data:
            if col in self.numeric_columns:
                # 数值型数据处理
                values = []
                for x in data[col]:
                    try:
                        # 移除引号并尝试转换为float
                        x = x.strip('"').strip("'")
                        val = float(x) if x != '' else None
                        values.append(val)
                    except (ValueError, AttributeError):
                        values.append(None)

                valid_values = [v for v in values if v is not None]
                mean_value = sum(valid_values) / len(valid_values) if valid_values else 0

                # 用平均值填充None值
                data[col] = [mean_value if v is None else v for v in values]
            else:
                # 分类型用众数填充
                value_counts = {}
                for x in data[col]:
                    x = str(x).strip('"').strip("'")
                    if x != '':
                        value_counts[x] = value_counts.get(x, 0) + 1
                mode = max(value_counts.items(), key=lambda x: x[1])[0] if value_counts else ''
                data[col] = [x.strip('"').strip("'") if x != '' else mode for x in data[col]]
        return data

    def _encode_categorical_variables(self, data):
        """编码分类变量"""
        for col in self.categorical_columns:
            if col in data:
                # 获取唯一值并排序
                unique_values = sorted(set(str(x).strip('"').strip("'") for x in data[col]))
                # 创建值到索引的映射
                value_to_index = {val: idx for idx, val in enumerate(unique_values)}
                # 编码
                data[col] = [value_to_index[str(val).strip('"').strip("'")] for val in data[col]]
        return data

    def _standardize_numeric_features(self, data):
        """标准化数值特征"""
        for col in self.numeric_columns:
            if col in data and col != self.target_column:  # 不对目标变量标准化
                values = [float(x) for x in data[col]]
                mean = sum(values) / len(values)
                std = (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5
                if std != 0:
                    data[col] = [(float(x) - mean) / std for x in data[col]]
                else:
                    data[col] = [0] * len(data[col])
        return data

    def split_features_target(self, data):
        """分割特征和目标变量"""
        features = {}
        target = None

        for col in data:
            if col != self.target_column:
                features[col] = data[col]
            else:
                target = data[col]

        return features, target

    def get_feature_names(self, data):
        """获取特征名称列表"""
        return [col for col in data.keys() if col != self.target_column]

    def convert_dict_to_matrix(self, data_dict):
        """将字典形式的数据转换为矩阵形式"""
        feature_names = self.get_feature_names(data_dict)
        n_samples = len(data_dict[feature_names[0]])
        n_features = len(feature_names)

        # 创建矩阵
        matrix = [[0] * n_features for _ in range(n_samples)]

        # 填充数据
        for i in range(n_samples):
            for j, feature in enumerate(feature_names):
                matrix[i][j] = float(data_dict[feature][i])

        return matrix