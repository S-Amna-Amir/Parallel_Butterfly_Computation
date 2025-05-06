/*#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <regex>
#include <vector>

using namespace std;

int main() {
    ifstream inputFile("wikiElec.ElecBs3.txt");
    ofstream outputFile("wiki_elec_extracted.txt");

    if (!inputFile.is_open()) {
        cerr << "Error opening input file!" << endl;
        return 1;
    }

    if (!outputFile.is_open()) {
        cerr << "Error opening output file!" << endl;
        inputFile.close();
        return 1;
    }

    string line;
    string current_user_id;
    regex user_regex(R"(U\s+(\d+))");
    regex vote_regex(R"(V\s+[-]?1\s+(\d+))"); // Matches V followed by a vote (-1, 0, or 1) and then the user ID

    while (getline(inputFile, line)) {
        smatch user_match;
        if (regex_search(line, user_match, user_regex) && user_match.size() > 1) {
            current_user_id = user_match[1];
        }

        smatch vote_match;
        if (regex_search(line, vote_match, vote_regex) && vote_match.size() > 1 && !current_user_id.empty()) {
            outputFile << current_user_id << " " << vote_match[1] << endl;
        }
    }

    inputFile.close();
    outputFile.close();

    cout << "Extracted vertex pairs and saved to wiki_elec_extracted.txt" << endl;

    return 0;
}*/



#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <regex>
#include <set>
#include <algorithm>

using namespace std;

int main() {
    ifstream inputFile("wikiElec.ElecBs3.txt");
    ofstream outputFile("wiki_elec_undirected.txt");

    if (!inputFile.is_open()) {
        cerr << "Error opening input file!" << endl;
        return 1;
    }

    if (!outputFile.is_open()) {
        cerr << "Error opening output file!" << endl;
        inputFile.close();
        return 1;
    }

    string line;
    string current_user_id;
    regex user_regex(R"(U\s+(\d+))");
    regex vote_regex(R"(V\s+1\s+(\d+))"); // Assuming you still want only support votes

    set<pair<string, string>> undirected_edges;

    while (getline(inputFile, line)) {
        smatch user_match;
        if (regex_search(line, user_match, user_regex) && user_match.size() > 1) {
            current_user_id = user_match[1];
        }

        smatch vote_match;
        if (regex_search(line, vote_match, vote_regex) && vote_match.size() > 1 && !current_user_id.empty()) {
            string u = current_user_id;
            string v = vote_match[1];

            // Ensure consistent ordering for undirected edges
            if (u > v) swap(u, v);
            undirected_edges.insert({u, v});
        }
    }

    for (const auto& edge : undirected_edges) {
        outputFile << edge.first << " " << edge.second << endl;
    }

    inputFile.close();
    outputFile.close();

    cout << "Extracted undirected vertex pairs (without doubling) and saved to wiki_elec_undirected.txt" << endl;

    return 0;
}
