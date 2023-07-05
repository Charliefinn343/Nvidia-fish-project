#!/usr/bin/python3
import jetson.inference
from jetson.inference import imageNet
import jetson.utils
import random

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str, help="filename of the image to process")
parser.add_argument("--network", type=str, default="googlenet", help="model to use, can be:  googlenet, resnet-18, ect. (see --help for others)")
opt = parser.parse_args()
img = jetson.utils.loadImage(opt.filename)
#net = jetson.inference.imageNet(opt.network)
net = imageNet(model="resnet18.onnx", labels="labels.txt", 
                input_blob="input_0", output_blob="output_0")
class_idx, confidence = net.Classify(img)
class_desc = net.GetClassDesc(class_idx)
print("image is recognized as '{:s}' (class #{:d}) with {:f}% confidence".format(class_desc, class_idx, confidence * 100))




list_of_fish = ["stripedbass", "bluegill", "whiteperch"]

# class_desc = random.choice(list_of_fish)
# print(class_desc)
print("")
###########################################################
facts_about_striped_bass = [
    "The Striped Bass, also known as Morone saxatilis, is a predatory fish found along the East Coast.",
    "It is recognized by its distinctive pattern of horizontal crosshatched lines on its body.",
    "Striped Bass can grow to impressive sizes, with some individuals reaching 5 to 6 feet in length and weighing between 77 to 125 pounds.",
    "On average, fully-grown Striped Bass are around two to three feet in length and weigh between 10 to 30 pounds.",
    "These fish have a lifespan ranging from 10 to 30 years.",
    "Striped Bass undergo three life stages: larvae, juvenile, and adult.",
    "Male Striped Bass reach sexual maturity at around 2 to 3 years of age, while females take longer, typically around 4 to 8 years.",
    "Breeding season for Striped Bass occurs from April to early June, and females lay their eggs near the shore in brackish or freshwater areas.",
    "After fertilization, adult Striped Bass leave the eggs to hatch on their own, and the larvae grow close to their birthplace during the summer.",
    "Striped Bass primarily inhabit rivers and the ocean near the coast, and they migrate between freshwater and saltwater for breeding.",
    "Their diet is opportunistic, consisting of anchovies, menhadens, crabs, squids, worms, and other small fish.",
    "Striped Bass face threats from larger fish, sharks, and fish-eating birds that prey on them.",
    "Human consumption poses a threat as well, as they are commonly caught for food.",
    "Climate change and pollution impact the population of Striped Bass, affecting their habitat and survival.",
    "Conservation efforts have played a vital role in the recovery of the Striped Bass population, which has now reached an above-average level."
]
facts_about_white_perch = [
    "White perch is a small, silvery fish with a dark, highly domed back.",
    "They are native to fresh and brackish waters in the Chesapeake Bay and its tidal tributaries.",
    "White perch typically grow 7 to 10 inches in length and rarely weigh more than 1 pound.",
    "They inhabit shallow, fresh and brackish waters, including quiet freshwater streams and channels.",
    "White perch can be found throughout the Chesapeake Bay and its rivers.",
    "Their diet consists of small fish, insects, detritus, and fish eggs and larvae.",
    "The average lifespan of white perch is up to 17 years, and their conservation status is stable.",
    "They have a silvery, greenish-gray body with faint lines on the sides, a whitish belly, and a highly domed, gray or blackish back.",
    "Predators of white perch include bluefish, weakfish, striped bass, and humans who catch them recreationally.",
    "White perch are semi-anadromous, with spawning runs occurring in late March as water temperatures increase.",
    "Each Chesapeake Bay river likely has its own population of white perch, as they do not venture far from their birthplace.",
    "The Maryland Chesapeake Bay record for white perch is a fish caught in 1979 weighing 2 pounds, 10 ounces."
]
facts_about_bluegill = [
    "Bluegill (Lepomis macrochirus) is a freshwater fish with an oval-shaped body that is deep and highly compressed.",
    "They have an olive green upper body and a light yellow belly, with dark bands running vertically from the back to the belly.",
    "Bluegills have blue and purple iridescence on their cheeks and a dark blue or black 'ear' on an extension of the gill cover.",
    "Breeding males may exhibit additional blue and orange coloration on their flanks.",
    "They are found inshore from the Great Lakes to Florida and in all tributaries of the Chesapeake Bay with salinity less than 18 ppt.",
    "Bluegills are typically about 6 inches in size but can reach up to 12 inches.",
    "They prefer quiet waters such as lakes, ponds, and slow-flowing rivers and streams.",
    "Spawning occurs from April to September in freshwater, where males create nests and guard the eggs and young.",
    "Bluegills often build their nests around other bluegill nests, creating honeycomb-like structures.",
    "Popular bait for bluegill fishing includes earthworms and corn kernels.",
    "Bluegills are known as 'sunnies' and are a member of the sunfish family.",
    "They are easy to catch, making them a favorite among young anglers."
]



facts = []

if class_desc == "stripedbass":
  facts += random.sample(facts_about_striped_bass, 4)

# Print the updated list
  # print(facts)
  for i in range(len(facts)):
    print(f"{i+1}. {facts[i]}\n")

if class_desc == "whiteperch":
  facts += random.sample(facts_about_white_perch, 4)

# Print the updated list
  # print(facts)
  for i in range(len(facts)):
    print(f"{i+1}. {facts[i]}\n")

if class_desc == "bluegill":
  facts += random.sample(facts_about_bluegill, 4)

# Print the updated list
  # print(facts)
  for i in range(len(facts)):
    print(f"{i+1}. {facts[i]}\n")





