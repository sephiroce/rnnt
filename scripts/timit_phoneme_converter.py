#-*- coding:utf-8 -*-


""" timit_phoneme_converter.py: It maps Timit 61 phones to 39 phones.
                                    aa, ao|aa
                              ah, ax, ax-h|ah
                                   er, axr|er
                                    hh, hv|hh
                                    ih, ix|ih
                                     l, el|l
                                     m, em|m
                                 n, en, nx|n
                                   ng, eng|ng
                                    sh, zh|sh
                                    uw, ux|uw
pcl, tcl, kcl, bcl, dcl, gcl, h#, pau, epi|sil
                                         q|-
"""
import json
import sys

__author__ = "Kyungmin Lee"
__email__ = "sephiroce@snu.ac.kr"

mapping_table = dict()
mapping_table["ao"] = "aa"

mapping_table["ax"] = "ah"
mapping_table["ax-h"] = "ah"

mapping_table["axr"] = "er"

mapping_table["hv"] = "hh"

mapping_table["ix"] = "ih"

mapping_table["el"] = "l"

mapping_table["em"] = "m"

mapping_table["en"] = "n"
mapping_table["nx"] = "n"

mapping_table["eng"] = "ng"

mapping_table["zh"] = "sh"

mapping_table["ux"] = "uw"

mapping_table["pcl"] = "sil"
mapping_table["tcl"] = "sil"
mapping_table["kcl"] = "sil"
mapping_table["bcl"] = "sil"
mapping_table["dcl"] = "sil"
mapping_table["gcl"] = "sil"
mapping_table["<s>"] = "sil"
mapping_table["h#"] = "sil"
mapping_table["pau"] = "sil"
mapping_table["epi"] = "sil"

mapping_table["q"] = "-"

phone_set = set()
with open(sys.argv[1]) as json_file:
  for line in json_file:
    json_data = json.loads(line.strip())
    new_text = ""
    for phone in json_data["text"].strip().split(" "):
      new_phone = mapping_table[phone] if phone in mapping_table else phone
      if new_phone == "-":
        continue
      phone_set.add(new_phone)
      new_text += new_phone + " "
    json_data["text"] = new_text.strip()
    print(json_data)
print(phone_set)
print(len(phone_set))
