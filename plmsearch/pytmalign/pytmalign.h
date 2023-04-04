#ifndef PYTMALIGN_H
#define PYTMALIGN_H

namespace python_tmalign {
    class pytmalign {
        public:
            pytmalign();
            ~pytmalign();
            double get_score(char * structure1, char * structure2);
    };
}

#endif