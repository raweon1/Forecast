import click
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

from source.Topic.CLI.Tree import *
from source.Topic.Model import *


def bar(root, threshold, threshold2, cp):
    new_child = False
    leaves = root.get_leaves()
    for leaf in leaves:
        a, b, c = cp.cond_prob_all_tokens(leaf.cum_name(), return_probabilities=True, return_absolut=True)
        children = a[np.intersect1d(np.where(b > threshold)[0], np.where(c > threshold2)[0], assume_unique=True)]
        for child in children:
            leaf.add_child(child)
            new_child = True
    return new_child


@click.command()
@click.argument('path', type=click.Path(exists=True))
def touch(path):
    """ """
    df = pd.read_csv(path)

    samp_size = len(df.review)
    sentences, token_lists, idx_in = preprocess(df.review, samp_size=samp_size)
    model = SentenceTransformer("distiluse-base-multilingual-cased")

    words, words_counts = np.unique(np.concatenate(token_lists), return_counts=True)
    embeddings = model.encode(words)

    click.echo("Creating synonyms")
    syn = Synonym(words, words_counts, embeddings, verbose=True)
    click.echo("Calculating probabilities")
    cp = ConditionalProbability(sentences, token_lists, syn)

    click.echo("Creating tree")
    root = Node(778)
    threshold = 0.1
    threshold2 = 5
    threshold_rate = 0.05
    while bar(root, threshold, threshold2, cp):
        threshold += threshold_rate

    click.pause("Press any Key to continue")

    click.clear()
    interactive(root, syn, cp)


def interactive(root, syn, cp):
    print_node(root, syn, cp)


def print_node(node, syn, cp):
    value = ""
    index = 0
    sentences = cp.sentences_from_tokens(node.cum_name())
    while value != "1":
        click.echo("Node: %s" % ",".join([syn.get_word(i) for i in node.cum_name()]))
        if not node.is_root():
            click.echo("Siblings: %s" % ",".join([syn.get_word(i.value) for i in node.parent.children if i != node]))
        click.echo("Children: %s" % ",".join([syn.get_word(i.value) for i in node.children]))
        click.echo()
        click.echo("Example text %d: " % (index + 1))
        click.echo(click.style(cp.sentences[sentences[index]], fg="red"))
        click.echo()
        click.echo("Please select a child to visit, 0 for visiting parent, 1 for quitting, 2 for next example")
        choices = [syn.get_word(child.value) for child in node.children] + ["0", "1", "2"]
        value = click.prompt("", type=click.Choice(choices, case_sensitive=False), show_choices=True)
        if value != "1":
            if value != "2":
                if value == "0" and not node.is_root():
                    node = node.parent
                elif value != "0":
                    node = node.get_child(syn.get_syn(value))
                sentences = cp.sentences_from_tokens(node.cum_name())
                index = 0
            else:
                index += 1
                index %= len(sentences)
        click.clear()


if __name__ == "__main__":
    touch()
