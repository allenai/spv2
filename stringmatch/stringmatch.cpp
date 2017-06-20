#include <stdlib.h>
#include <limits.h>

/* start_pos is inclusive, end_pos is exclusive. */
struct MatchResult {
    int start_pos;
    int end_pos;
    int cost;
};


class IntMatrix {
    unsigned int nrows;
    unsigned int ncols;
    int* data;

  public:
    IntMatrix(unsigned int nrows, unsigned int ncols) {
      this->nrows = nrows;
      this->ncols = ncols;
      this->data = new int[nrows*ncols];
    }

    ~IntMatrix() {
      delete data;
    }

    int* operator[](const int x) {
      return &data[x * ncols];
    }

    int& operator()(const int x, const int y) { return data[x * ncols + y]; }
};


/*
 * Find the location of the substring in text with the minimum edit distance
 * (Levenshtein) to key. Given a key of length n and text of length m, we can
 * do this in O(n*m) time using dynamic programming.
 */
MatchResult match(const wchar_t* key, const wchar_t* text) {
  const int key_len = wcslen(key);
  const int text_len = wcslen(text);

  MatchResult res;

  IntMatrix distance = IntMatrix(key_len+1, text_len+1);
  IntMatrix start_pos = IntMatrix(key_len+1, text_len+1);
  //int distance[key_len+1][text_len+1];
  //int start_pos[key_len+1][text_len+1];

  int key_idx;
  int text_idx;
  
  // Allow the match to start anywhere along the text
  for (text_idx = 0; text_idx < text_len + 1; text_idx++) {
    distance[0][text_idx] = 0;
    start_pos[0][text_idx] = text_idx;
  }
  
  for (key_idx = 1; key_idx < key_len + 1; key_idx++) {
    distance[key_idx][0] = distance[key_idx-1][0] + 1;
    start_pos[key_idx][0] = 0;
    for (text_idx = 1; text_idx < text_len + 1; text_idx++) {
      int added_in_key = distance[key_idx-1][text_idx] + 1;
      int added_in_text = distance[key_idx][text_idx-1] + 1;
      int substitute = distance[key_idx-1][text_idx-1];
      if (text[text_idx-1] != key[key_idx-1]) {
        substitute += 1;
      }
      int cur_dist = added_in_key;
      int cur_start = start_pos[key_idx-1][text_idx];
      if (added_in_text < cur_dist) {
        cur_dist = added_in_text;
        cur_start = start_pos[key_idx][text_idx-1];
      }
      if (substitute < cur_dist) {
        cur_dist = substitute;
        cur_start = start_pos[key_idx-1][text_idx-1];
      }
      distance[key_idx][text_idx] = cur_dist;
      start_pos[key_idx][text_idx] = cur_start;
    }
  }

  int best_dist = INT_MAX;
  int best_start_pos = -1;
  int best_end_pos = -1;
  for (text_idx = 1; text_idx < text_len + 1; text_idx++){
    int cur_dist = distance[key_len][text_idx];
    if (cur_dist < best_dist) {
      best_dist = cur_dist;
      best_start_pos = start_pos[key_len][text_idx];
      best_end_pos = text_idx - 1;
    }
  }

//  free(distance);
//  free(start_pos);

  res.start_pos = best_start_pos;
  res.end_pos = best_end_pos + 1; // end_pos is exclusive
  res.cost = best_dist;
  return res;
}

