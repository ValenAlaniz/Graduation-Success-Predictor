import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import KBinsDiscretizer
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


def cargar_datos():
    with open('data.csv', newline='', encoding="utf-8") as corpus_csv:
        corpus_df = pd.read_csv(corpus_csv, sep=";")
        return corpus_df


def discretizar_variable_kmeans(X_df, column_to_discretize, cant_bins):
    X_to_discretize = X_df[[column_to_discretize]]
    kbd = KBinsDiscretizer(n_bins=cant_bins, encode='ordinal', strategy='kmeans')
    kbd.fit(X_to_discretize)
    X_binned = kbd.transform(X_to_discretize)
    df_binned = pd.DataFrame(X_binned, columns=[column_to_discretize])
    # Fusionar el DataFrame original con las columnas discretizadas
    X_df[column_to_discretize] = df_binned
    return X_df


def preprocessing_training_set(X_df):
    X_df['Target'] = X_df['Target'].apply(lambda x: 'Graduate' if x != 'Dropout' else x)

    X_df = discretizar_variable_kmeans(X_df, 'Previous qualification (grade)', 3)
    X_df = discretizar_variable_kmeans(X_df, 'Admission grade', 3)
    X_df = discretizar_variable_kmeans(X_df, 'Unemployment rate', 5)
    X_df = discretizar_variable_kmeans(X_df, 'Inflation rate', 5)
    X_df = discretizar_variable_kmeans(X_df, 'GDP', 2)
    X_df = discretizar_variable_kmeans(X_df, 'Curricular units 2nd sem (grade)', 3)
    X_df = discretizar_variable_kmeans(X_df, 'Curricular units 1st sem (grade)', 3)
    X_df = discretizar_variable_kmeans(X_df, 'Age at enrollment', 10)

    return X_df


def calc_entropy(column):
    counts = column.value_counts()
    p = counts / counts.sum()
    entropy = -np.sum(p * np.log2(p))

    return entropy


def calc_gain(df, entropyX, attribute_column):
    attribute_values = df[attribute_column].unique()
    total_samples = len(df)

    final_entropy = 0

    for value in attribute_values:
        df_filtered = df[df[attribute_column] == value]
        df_filtered_entropy = calc_entropy(df_filtered['Target'])
        final_entropy += (len(df_filtered) / total_samples) * df_filtered_entropy

    gain = entropyX - final_entropy

    return gain


def split_information(df, attribute_column):
    attribute_values = df[attribute_column].unique()
    total_samples = len(df)
    split_info = 0
    for value in attribute_values:
        df_filtered = df[df[attribute_column] == value]
        aux = len(df_filtered) / total_samples
        split_info += aux * np.log2(aux)

    return -split_info


def calc_gain_ratio(df, entropyX, attribute_column):
    gain = calc_gain(df, entropyX, attribute_column)
    split_info = split_information(df, attribute_column)

    if split_info == 0:
        return 0

    return gain / split_info


def get_best_attribute(df, Y_df, min_split_gain, use_gain_ratio):
    gains = {}
    entropyX = calc_entropy(Y_df)
    X_df = df.drop('Target', axis=1)

    for attribute in X_df.columns.tolist():
        if use_gain_ratio:
            gains[attribute] = calc_gain_ratio(df, entropyX, attribute)
        else:
            gains[attribute] = calc_gain(df, entropyX, attribute)

    # guardo la clave de la entrada de gains con mayor ganancia
    best_attr = max(gains, key=gains.get)
    if gains[best_attr] < min_split_gain:
        return None
    else:
        return best_attr


def add_node_tree(tree, id_node, id_parent, tag, edge_value=None):
    new_row = {"node": id_node, "tag": tag, "parent": id_parent, "children": []}
    # Busco el nodo padre y le agrego el id del nodo hijo
    for row in tree:
        if row["node"] == id_parent:
            row["children"].append([id_node, edge_value])

    tree.append(new_row)
    return tree


def evaluate_instance(instance, tree):
    # Busca el nodo raíz (el nodo cuyo id_parent es -1)
    root = next(node for node in tree if node["parent"] == -1)

    # Comienza el recorrido desde la raíz
    current_node = root

    while current_node["tag"] not in ["Graduate", "Dropout"]:
        attribute = current_node["tag"]
        value = instance[attribute]

        exists_child = False

        # Busca el hijo que corresponde al valor del atributo en la instancia
        for child_id, edge_value in current_node["children"]:
            if edge_value == value:
                # Encuentra el nodo hijo en el árbol
                current_node = next(node for node in tree if node["node"] == child_id)
                exists_child = True
                break  # Sal del bucle una vez que encuentres el nodo hijo apropiado

        if not exists_child:
            # Si no existe el hijo continuo por el nodo que se parezca más al valor del atributo
            best_node = None
            best_value = None

            for child_id, edge_value in current_node["children"]:
                if best_node is None or abs(edge_value - value) < abs(best_value - value):
                    best_node = child_id
                    best_value = edge_value

            current_node = next(node for node in tree if node["node"] == best_node)

    return current_node["tag"]  # Devuelve la etiqueta de la hoja a la que llegaste


def evaluate_dataset(X_df, tree):
    results = []

    for index, row in X_df.iterrows():
        results.append(evaluate_instance(row, tree))

    return results


def get_confusion_matrix(y_true, y_pred):
    # Calcular la matriz de confusión
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Etiquetas para las clases
    class_names = [0, 1]

    # Crear la figura y el eje
    plt.figure(figsize=(8, 6))

    # Crear un mapa de calor
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)

    # Etiquetas y títulos
    plt.ylabel('Clase Verdadera')
    plt.xlabel('Clase Predicha')
    plt.title('Matriz de Confusión')

    # Mostrar la trama
    plt.show()


node_id = 0  # Global node_id


def ID3(tree, parent, X_df, Y_df, elem, min_samples_split, min_split_gain, use_gain_function):
    global node_id

    unique_value = Y_df.unique()

    if len(unique_value) == 1:
        tree = add_node_tree(tree, node_id, parent, unique_value[0], elem)
        node_id += 1
        return tree

    selected_attribute = get_best_attribute(X_df, Y_df, min_split_gain, use_gain_function)

    if len(unique_value) == 0:
        tag = 'Graduate'
        tree = add_node_tree(tree, node_id, parent, tag, elem)
        node_id += 1
        return tree

    if selected_attribute is None or len(X_df) < min_samples_split:
        tag = Y_df.value_counts().idxmax()
        tree = add_node_tree(tree, node_id, parent, tag, elem)
        node_id += 1
        return tree

    tree = add_node_tree(tree, node_id, parent, selected_attribute, elem)
    parent = node_id
    node_id += 1

    for elem in X_df[selected_attribute].unique():
        X_df.loc[:, 'Target'] = Y_df
        X_df_filtered = X_df[X_df[selected_attribute] == elem]
        y_df_filtered = X_df_filtered['Target']
        ID3(tree, parent, X_df_filtered, y_df_filtered, elem, min_samples_split, min_split_gain, use_gain_function)
