#! -*- coding: utf-8 -*-
# Author: DengBoCong <bocongdeng@gmail.com>
#
# License: Apache License 2.0

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import os
import sys
import pickle
import jieba
import requests
from PyQt6.QtCore import QMetaObject, QCoreApplication, Qt, QModelIndex
from PyQt6.QtGui import QStandardItemModel, QKeyEvent
from PyQt6.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QVBoxLayout, QLabel, QTableView, QPushButton,
    QRadioButton, QLineEdit, QAbstractItemView, QTreeWidget, QCheckBox, QHeaderView,
    QSplitter, QFrame, QTextBrowser, QListView, QTreeWidgetItem, QListWidget, QListWidgetItem
)
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Dict, List, Tuple


class SearchEngine(object):
    def __init__(self):
        super(SearchEngine, self).__init__()
        self.stopwords = self.get_stopwords()
        self.index = {"title": {}, "desc": {}}
        self.document = {}
        self.categories = {}

        self.create_index()

    @staticmethod
    def get_stopwords(file_path: str = "./paper-code/stopwords/stopwords.pkl"):
        with open(file_path, "rb") as file:
            return pickle.load(file)

    def filter(self, text: str):
        tokens = [token.lower() for token in jieba.cut(text) if token.strip() not in self.stopwords]
        return tokens

    def create_index(self, readme_file_path: str = "./README.md"):
        index_count = 100000000
        with open(readme_file_path, "r", encoding="utf-8") as file:
            for line in file:
                if line.startswith("+ ["):
                    line = line.strip().strip("\n").split("|")
                    if len(line) < 4:
                        continue
                    paper_info = re.findall(r"\[(.*)\]\((.*)\)", line[1])
                    publish_info = line[-1].split(",")
                    categories = line[0].strip()[3:-1].split("-")
                    paper_desc = line[-2].strip()
                    paper_title = paper_info[0][0].strip()
                    paper_read_info = []
                    for start in range(2, len(line) - 2):
                        temp_info = re.findall(r"\[(.*)\]\((.*)\)", line[start])
                        paper_read_info.extend(temp_info)

                    # "paper_read_link": re.findall(r"\((.*)\)", line[2])[0][0].strip() if len(line) == 5 else "",
                    paper = {
                        "paper_index": index_count,
                        "paper_title": paper_title,
                        "paper_link": paper_info[0][1].strip(),
                        "paper_categories": categories,
                        "paper_year": publish_info[-1].strip(),
                        "paper_author": publish_info[0].strip(),
                        "paper_desc": paper_desc,
                        "paper_read": paper_read_info
                    }
                    self.document[index_count] = paper

                    for keyword in self.filter(paper_title):
                        if keyword not in self.index["title"]:
                            self.index["title"][keyword] = []
                        self.index["title"][keyword].append(index_count)
                    for keyword in self.filter(paper_desc):
                        if keyword not in self.index["desc"]:
                            self.index["desc"][keyword] = []
                        self.index["desc"][keyword].append(index_count)

                    for category in categories:
                        if category not in self.categories:
                            self.categories[category] = []
                        self.categories[category].append(self.document[index_count])

                    index_count += 1

    def search(self, query: str, search_type: str):
        keywords, papers = self.filter(query), set()
        for keyword in keywords:
            if search_type in ["title", "all"] and keyword in self.index["title"]:
                papers.update([index_id for index_id in self.index["title"][keyword]])
            if search_type in ["desc", "all"] and keyword in self.index["desc"]:
                papers.update([index_id for index_id in self.index["desc"][keyword]])

        return [self.document[paper_index] for paper_index in papers], keywords


class SearchKits(object):
    def __init__(self, width: int = 1118, height: int = 520):
        self.app = QApplication(sys.argv)
        self.windows = QWidget()
        self.thread_pool = ThreadPoolExecutor(max_workers=5)
        self.search_engine = SearchEngine()

        if not self.windows.objectName():
            self.windows.setObjectName("windows")
        self.windows.setFixedSize(width, height)
        self.base_layout = QHBoxLayout(self.windows)
        self.base_layout.setObjectName("base_layout")
        self.left_vertical_layout = QVBoxLayout()
        self.left_vertical_layout.setSpacing(6)
        self.left_vertical_layout.setObjectName("left_vertical_layout")

        self.category_label = QLabel(self.windows)
        self.category_label.setObjectName("category_label")
        self.category_label.setText("Paper Category List")
        self.category_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.left_vertical_layout.addWidget(self.category_label)

        self.left_paper_tree = QTreeWidget(self.windows)
        self.left_paper_tree.setObjectName("paper_tree_view")
        self.left_paper_tree.setColumnCount(2)
        self.left_paper_tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        self.left_paper_tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        self.left_paper_tree.setColumnWidth(0, 370)
        self.left_paper_tree.setColumnWidth(1, 40)
        self.left_paper_tree.setHeaderLabels(["Categories", "Year"])
        self.left_paper_tree.setSelectionMode(QTreeWidget.SelectionMode.ExtendedSelection)
        self.left_paper_tree.header().setDefaultAlignment(Qt.AlignmentFlag.AlignCenter)
        self.left_vertical_layout.addWidget(self.left_paper_tree)

        self.right_vertical_layout = QVBoxLayout()
        self.right_vertical_layout.setObjectName("right_vertical_layout")

        self.search_slide_label = QLabel(self.windows)
        self.search_slide_label.setObjectName("search_slide_label")
        self.search_slide_label.setText(" ")
        self.search_slide_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.right_vertical_layout.addWidget(self.search_slide_label)

        self.checkbox_horizontal_layout = QHBoxLayout()
        self.checkbox_horizontal_layout.setObjectName("checkbox_horizontal_layout")

        self.paper_title_radio_button = QRadioButton(self.windows)
        self.paper_title_radio_button.setObjectName("paper_title_radio_button")
        self.paper_title_radio_button.setText("Paper Title")
        self.checkbox_horizontal_layout.addWidget(self.paper_title_radio_button)

        self.paper_desc_radio_button = QRadioButton(self.windows)
        self.paper_desc_radio_button.setObjectName("paper_desc_radio_button")
        self.paper_desc_radio_button.setText("Paper Desc")
        self.checkbox_horizontal_layout.addWidget(self.paper_desc_radio_button)

        self.paper_all_radio_button = QRadioButton(self.windows)
        self.paper_all_radio_button.setObjectName("paper_all_radio_button")
        self.paper_all_radio_button.setText("Global Search")
        self.checkbox_horizontal_layout.addWidget(self.paper_all_radio_button)

        self.input_search_horizontal_layout = QHBoxLayout()
        self.input_search_horizontal_layout.setObjectName("input_search_horizontal_layout")

        self.input_search_label = QLabel(self.windows)
        self.input_search_label.setObjectName("input_search_label")
        self.input_search_label.setText("Input Keyword: ")
        self.input_search_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.input_search_horizontal_layout.addWidget(self.input_search_label)

        self.keyword_line_edit = QLineEdit(self.windows)
        self.keyword_line_edit.setObjectName("keyword_line_edit")
        self.keyword_line_edit.setPlaceholderText("Default global search..")
        self.keyword_line_edit.returnPressed.connect(self.search_paper)
        self.input_search_horizontal_layout.addWidget(self.keyword_line_edit)

        self.search_pushbutton = QPushButton(self.windows)
        self.search_pushbutton.setObjectName("search_pushbutton")
        self.search_pushbutton.setText("search")
        self.search_pushbutton.clicked.connect(self.search_paper)
        self.input_search_horizontal_layout.addWidget(self.search_pushbutton)

        self.search_res_browser = QTextBrowser(self.windows)
        self.search_res_browser.setObjectName("search_res_browser")

        self.right_vertical_layout.addLayout(self.checkbox_horizontal_layout)
        self.right_vertical_layout.addLayout(self.input_search_horizontal_layout)
        self.right_vertical_layout.addWidget(self.search_res_browser)

        self.base_layout.addLayout(self.left_vertical_layout)
        self.base_layout.addLayout(self.right_vertical_layout)
        self.base_layout.setStretch(0, 2)
        self.base_layout.setStretch(1, 3)

    def set_paper_category_list(self, categories_papers: Dict[str, List[Dict[str, Any]]]):
        for index, (category_name, papers) in enumerate(categories_papers.items()):
            setattr(self, f"category_item_{index}", QTreeWidgetItem(self.left_paper_tree))
            getattr(self, f"category_item_{index}").setText(0, category_name)
            getattr(self, f"category_item_{index}").setText(1, "")
            for paper_index, paper_info in enumerate(papers):
                setattr(self, f"paper_{index}_{paper_index}", QTreeWidgetItem(getattr(self, f"category_item_{index}")))
                getattr(self, f"paper_{index}_{paper_index}").setText(0, paper_info["paper_title"])
                getattr(self, f"paper_{index}_{paper_index}").setText(1, paper_info["paper_year"])
                getattr(self, f"paper_{index}_{paper_index}").setText(2, str(paper_info["paper_index"]))

        self.left_paper_tree.clicked.connect(self.click_item_action)

    def set_search_res_listview(self, paper_res: List[Dict[str, str]],
                                start_content: str = "", keywords: List[str] = None, search_type: str = "all"):
        content = start_content
        for paper in paper_res:
            paper_title = paper["paper_title"]
            paper_desc = paper['paper_desc']
            if keywords is not None:
                for keyword in keywords:
                    reg = re.compile(re.escape(keyword), re.IGNORECASE)
                    if search_type in ["all", "title"]:
                        paper_title = reg.sub(f'<font color="red">{keyword.upper()}</font>', paper["paper_title"])
                    if search_type in ["all", "desc"]:
                        paper_desc = reg.sub(f'<font color="red">{keyword.upper()}</font>', paper["paper_desc"])
            content += f"#### {paper_title}\n"
            content += f"+ Author: {paper['paper_author']}\n"
            content += f"+ Year: {paper['paper_year']}\n"
            content += f"+ Tag: {', '.join(paper['paper_categories'])}\n"
            content += f"+ Link: {paper['paper_link']}\n"
            for read_info in paper["paper_read"]:
                content += f"+ {read_info[0]}: {read_info[1]}"
            content += "\n"
            content += f"+ Sketch: {paper_desc}\n"

            content += "---\n"

        self.search_res_browser.setOpenExternalLinks(True)
        self.search_res_browser.setMarkdown(content)

    def search_paper(self):
        search_type = "all"
        if self.paper_title_radio_button.isChecked():
            search_type = "title"
        if self.paper_desc_radio_button.isChecked():
            search_type = "desc"

        start_time = datetime.now().timestamp()
        paper_res, keywords = self.search_engine.search(query=self.keyword_line_edit.text(), search_type=search_type)
        cost_time = datetime.now().timestamp() - start_time
        self.set_search_res_listview(
            paper_res, "   *Find %d results (%.6f seconds)*\n" % (len(paper_res), cost_time), keywords, search_type
        )

    def click_item_action(self, point: QModelIndex):
        if self.left_paper_tree.currentItem().text(2) != "":
            index_counts = []
            for select_item in self.left_paper_tree.selectedItems():
                if select_item.text(2) != "":
                    index_counts.append(select_item.text(2))

            paper_res = [self.search_engine.document[int(item_index)] for item_index in index_counts]
            self.set_search_res_listview(paper_res)

    def show(self, title: str = "Paper Search Tool"):
        self.set_paper_category_list(self.search_engine.categories)
        content = ""
        count = 0
        for key in self.search_engine.categories.keys():
            if count % 6 == 0:
                content += "<br>"
            count += 1
            content += f"•&nbsp;&nbsp;[{key}](https://github.com/DengBoCong/nlp-paper)&nbsp;&nbsp;"

        with open("./te.txt", "w", encoding="utf-8") as file:
            file.write(content)
        self.windows.show()
        self.windows.setWindowTitle(title)
        sys.exit(self.app.exec())


if __name__ == '__main__':
    kits = SearchKits()
    kits.show()
