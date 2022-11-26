# moai-challenge
Chest CT에서의 인공지능 기반 자동 Body morphometry 측정

## 🧑🏻‍💻Team members
- sangho Kim | [[github]](https://github.com/sangh0)
- yujin Kim | [[github]](https://github.com/yujinkim1)

## 🔨Dev environment
- [Python](https://www.python.org/downloads) 3.9.x
- [CUDA](https://developer.nvidia.com/cuda-toolkit) 11.x
- [CuDNN](https://developer.nvidia.com/cudnn)
- [Pytorch](https://pytorch.org/docs/stable/index.html)

## Team Convention
### Commit
- `type: subject` 형식으로 메세지를 작성합니다. 이때 spacing은 `:` 앞에 한 번만 사용합니다. 
- subject 작성 시 영어를 사용할 경우 반드시 대문자로 시작하고 이어지는 단어는 spacing 후 소문자로 작성합니다.
- 메세지를 끝맺을 때 `., !, -, ?`와 같은 마침표를 사용하지 않습니다.

    ```zsh
    #examples
    feat: "Format converter"
    add: "Reference urls"
    edit: "Model train function"
    test: "Model test"
    fix: "Unusual execution"
    refactor: "Delete duplicate code"
    style: "Add missing semicolons"
    ```

### Push&Pull guideline
1. 내용을 추가하거나 변경하는 작업을 수행하기 전에 `git pull {브랜치 이름}`을 먼저 수행해야합니다.
2. 다음으로 `git push {브랜치 이름}` 수행합니다.

### Branch guideline
1. **현재 브랜치 확인(Check branch)**
    ```zsh
    git branch
    git branch -v #상세 정보도 함께 확인
    ```
    >브랜치를 생성하거나 변경, 삭제하기 전에 로컬에서 어떤 브랜치를 가리키고 있는지 먼저 확인합니다.
2. **브랜치 생성(Create branch)**
    ```zsh
    git branch -b {생성할 브랜치 이름}
    ```
    >`-b` 옵션을 반영하면 브랜치 생성과 동시에 해당 브랜치로 변경할 수 있습니다. 
3. **브랜치 이동(Switch branch)**
    ```zsh
    git switch {이동할 브랜치 이름}
    ```
    >기존에는 `checkout` 명령어를 사용했지만, 현재는 `switch` 명령어를 사용해 브랜치를 변경합니다. 따라서 `checkout` 명령어는 사용하지 않습니다.
4. **원격저장소 반영(Remote branch)**
    ```zsh
    git push origin {반영할 브랜치 이름}
    ```
    >로컬에서 브랜치를 생성 후 원격 저장소에도 해당 브랜치를 반영해야합니다.
5. **브랜치 삭제하기(Delete branch)**
    ```zsh
    git branch -D {삭제할 브랜치 이름}
    ```
    > `-D` 옵션을 반영하면 브랜치를 삭제할 수 있습니다. 단, 해당 브랜치가 로컬에 존재해야합니다.
---
## ✨Update history

<details>
<summary>2022.11.21.</summary>
<div markdown="1">

- `create` private repository
- `create` git projects
- `add` collaborator
- `edit` README file

</div>
</details>

<details>
<summary>2022.11.26.</summary>
<div markdown="1">

- `create` dataset
- `remove` augment

</div>
</details>