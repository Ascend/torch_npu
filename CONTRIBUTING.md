# Table of Contents

- [Unit Testing](#unit-testing)
  - [C++ Unit Testing](#c-unit-testing)

## Unit Testing
### C++ Unit Testing

**Build Unit Test**

Set Environment Variable `BUILD_TEST=ON` before compiling source code, the executable will stored in path `./build/gtest`.

```bash
export BUILD_TEST=ON
python setup.py develop

# or 
BUILD_TEST=ON python setup.py develop
```

**Run Unit Test**

Tests using the GTest framework, you can run tests using following command after compiled.

```bash
cd build && ctest --output-on-failure --verbose
```
