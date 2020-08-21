import subprocess
from subprocess import PIPE
import os

TMP_FILENAME = "tmp_jwordsplitter"
JAR = "jwordsplitter.jar"


def split_words(word_list):
    with open(TMP_FILENAME, "w") as file:
        for word in word_list:
            file.write(word + "\n")
    out = subprocess.run(["C:/Program Files (x86)/Java/jre1.8.0_261/bin/java", "-jar", JAR, TMP_FILENAME],
                         shell=True,
                         stdout=PIPE)
    os.remove(TMP_FILENAME)
    return [word.split(", ") for word in out.stdout.decode("utf-8").splitlines()]


print(split_words(["Dachdeckerarbeiten", "Wohnraum", "Wohnrecht"]))
