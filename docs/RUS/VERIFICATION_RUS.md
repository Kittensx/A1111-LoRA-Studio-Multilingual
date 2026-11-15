# Проверка кандидата v15 — 2026-11-13

Тестовая база: локальный `stable-diffusion-webui-dev.zip` с наложением локального
`modules.zip`. Это не та же композиция базы, что в независимом Linux-аудите.
Новая копия: `<HISTORICAL_AUDIT_ROOT>/tmp/lora-v15-installed-kha0e4qa`.
Журнал: `evidence/v15-installed-smoke-accepted.log`.

- 142 Torch CPU runtime PASS, включая 8 новых asset-identity сценариев и
  отдельный тест пяти маршрутов (четыре ожидаемо НЕ поддержаны).
- 64 parser/API/config PASS + 13 subtests PASS.
- 22 Node PASS, 62 UI assertions; настоящий браузер/Gradio не запускался.
- Installer/header inspectors: 26 PASS, 1 SKIP (ОС не разрешает создание symlink).
- После установки 73/73 SAME; повторное планирование не требует изменений.
- Rollback 59 реально изменённых/добавленных путей; сравнение всех 73 целевых
  путей с первоначальными байтами/отсутствием успешно.
- 16 UNKNOWN допускаются ТОЛЬКО в этой новой одноразовой тестовой копии.
  Рабочая установка по-прежнему блокирует неизвестные версии без явного флага.
- Ruff изменённых Python и Node syntax PASS. Целостность manifest и старых
  релизов проверяет `verify_package_v15.py` в рабочем каталоге сборки.

Два промежуточных smoke-прогона остановились на UI-тестах после переименования
заголовков кривых. Стабильные aria-label сохранены, тест заголовка обновлён с
дополнительной проверкой семантики TE/UNet. Полный повторный прогон выше прошёл;
неуспешные промежуточные запуски не выдаются за успешную приёмку.

Оговорки: тесты используют production AST и реальные Torch tensors, но заменяют
отсутствующие WebUI-сервисы; новые cache-тесты используют dict вместо diskcache,
малый stdlib decoder вместо safetensors. Предупреждения об отсутствии NumPy в
Torch-venv и старом pytest `base_url` остаются. GPU/качество/VRAM не измерялись.
Точный SHA256 финального ZIP приводится во внешней квитанции v15.

---

# Проверка кандидата v14 — 2026-11-12

## Глубокая проверка LoRA cache

Добавлены реальные same-size replacement и file-changed-during-read сценарии для
LoRA payload cache. См. DEEP_AUDIT_V14_RU.md. v14 повторно проходит все проверки
установки/отката ниже; результаты текущего исходного проекта: 133 runtime,
62 parser/API/config (+13 subtests), 22 Node/61 UI assertions.

---

# Проверка кандидата v13 — 2026-11-12

## Дополнительный самостоятельный аудит

Добавлена проверка конкурентного data-only изменения одного Options key во время
долгого callback. Тест требует, чтобы callback применился только для первого
вызова, а второе изменение с `run_callbacks=False` не инициировало скрытую
перезагрузку. Подробности и границы: SELF_AUDIT_V13_RU.md.

v13 повторно проходит все нижеперечисленные проверки установки/отката.

---

# Проверка кандидата v12 — 2026-11-12

## Финальный прогон установленной копии

`tools/post_install_smoke.py` использует два уже имеющихся локальных архива,
устанавливает пакет обычным кодом install.py, затем копирует ТОЛЬКО test-файлы.
Runtime modules из reference/verification-only поверх установки не копируются.
Ничего не скачано; использованы существующие Torch CPU, pytest и Node.

- 131 Torch CPU/production-AST runtime PASS (в том числе 19 новых проверок аудита).
- 61 parser/API/config PASS, 13 subtests PASS.
- 22 Node PASS, 61 UI assertions в изолированном DOM shim.
- Повторный план установки: все 70 операций SAME.
- Rollback: восстановлены/убраны 56 изменённых файлов; проверены исходные байты
  всех затронутых путей, включая отсутствие добавленных файлов после отката.
- Сохранённая тестовая копия после отката:
  `<HISTORICAL_AUDIT_ROOT>/tmp\lora-v12-installed-xggld_54`.
- Установщик/inspector: 27 методов, 26 PASS / 1 SKIP (ОС запрещает symlink).
  Сообщение Installation interrupted в тесте намеренного сбоя ожидаемо.
- `install.py check` рабочего проекта: 0 changes, 0 unrecognized; 110 assets,
  70 операций. Отдельный тест отклоняет manifest без обязательной REGION-операции.
- Ruff изменённых Python runtime/tests и инструментов пакета PASS; это не полный
  lint репозитория. `inspect_embedding_header.py --help` запускается без Torch.

База: локальный Downloads/stable-diffusion-webui-dev.zip + source/modules.zip.
Её тождественность базе чужого аудита не предполагается. 16 UNKNOWN разрешены
ТОЛЬКО в новой тестовой копии; обычный установщик сохраняет защиту UNKNOWN.
При runtime-тестах ожидаемы предупреждение Torch об отсутствующей NumPy и
сообщения восстановления checkpoint в сценариях искусственного отказа.

Воспроизведение (из корня v12; указать свои существующие пути):

```text
python -B tools/post_install_smoke.py --base-zip BASE.zip --modules-zip MODULES.zip --temp-root <HISTORICAL_AUDIT_ROOT>/tmp --torch-python <HISTORICAL_AUDIT_ROOT>/venv/Scripts/python.exe
python -B -m unittest discover -s tests
python -B install.py verify
```

Системный Python для smoke должен уже иметь pytest/FastAPI; отдельный Python —
Torch. Скрипт не устанавливает их. Тестовая копия и backup сохраняются на D:.

Полный перечень подтверждений, изменений совместимости и ограничений:
EXTERNAL_AUDIT_RU.md. Реальный Gradio, CUDA/WAI, пользовательские TI/LoRA,
качество изображений и VRAM НЕ проверены этим прогоном. Браузерные 66 PASS ниже
относятся к v11 и не являются новым запуском браузера v12.

Ниже сохранены исторические результаты предыдущих версий.

---

# Проверка кандидата v11 — 2026-11-11

## Проверки исходного проекта v11

112 Torch CPU/production-AST runtime PASS; 59 parser/API/config PASS и 13 subtests;
22 Node PASS с 61 UI assertions. В настоящем браузере изолированный fixture дал
66 PASS, включая собственную ширину панели и контраст dark/light без theme tokens.
Это не полный сервер Gradio: fixture подменяет API, CPU-тесты — внешние сервисы.
Новая матрица 1/2/4 вхождений, base/Hires, 5 ключей, 11 шагов даёт 330 сценариев.
Подробности воспроизведений и ограничения: COMBINATIONS_AUDIT_RU.md.

## Повторный прогон из самого пакета v11

- 112 Torch CPU runtime PASS.
- 59 parser/API/config PASS, 13 subtests PASS.
- 22 Node PASS, включая 61 UI assertions в DOM shim.
- Установщик и inspector: 25 PASS, 1 SKIP (ОС не разрешает создание symlink).
  Включены полная установка/побайтовый rollback в синтетическом host на D:,
  отказ посередине установки и сохранение пользовательских правок при откате.
- `install.py verify`: 104 assets PASS. `check` рабочего проекта: 0 changes,
  0 unrecognized. 64 операции; область установки не расширена относительно v10.
- Ruff четырёх изменённых Python runtime и двух новых Python tests PASS.
  Это не полный lint репозитория; ESLint не устанавливался.
- modules payload, processing patch и lyco_helpers.py не менялись от v10.

Воспроизведение (из `files/`, существующие Node/Python/Torch/pytest):

```text
python -B -m unittest discover -s test -p "test_runtime_*.py"
node --test test/lora*.test.cjs
python -B -m pytest -q --noconftest -o addopts='' test/test_lora_interaction_contract.py test/test_config_store.py test/test_lora_studio_access.py test/test_lora_presets_transaction.py test/test_lora_advanced.py test/test_lora_schema.py test/test_extra_networks_parser.py test/test_lora_quick_presets.py
```

Для pytest здесь использовался `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`; Torch CPU
и pytest запущены разными уже существующими Python-средами. Из корня пакета:
`python -B -m unittest discover -s tests -v`, `python install.py verify`.
Для браузера: обслужить `files/` локальным HTTP server и открыть
`test/lora_studio_fixture.html`, нажать Run UI checks. Ничего не устанавливать
и не запускать публичный сервер ради этих проверок.

Ниже сохранена история v10 и более ранних пакетов.

# Проверка кандидата v10 — 2026-11-06

## Текущий прогон v10

- Исходный проект и затем файлы пакета: 104 runtime PASS, настоящий Torch CPU.
- Из пакета: 54 parser/API/config PASS (+13 subtests), 23 NumPy loader PASS.
- Финальный callback-lock regression: Options callback может дождаться другого
  потока, меняющего Options. Config lock не удерживается при ожидании GPU queue.
- Установщик/inspector: 26 методов, 25 PASS / 1 OS-symlink SKIP; включают
  синтетическую полную установку и откат. Сообщение Installation interrupted
  внутри failure-injection теста ожидаемо и не означает сбой реальной установки.
- Node: 17 PASS, 51 UI assertions в исходном проекте. Это DOM shim, не Gradio.
- Предыдущие внешние воспроизводители: runtime/VAE 6 PASS, recovery 5 PASS,
  reported LoRA loading 4 PASS. SD3 standalone 5 PASS (включая старый baseline).
- Ruff изменённых runtime/tests, кроме processing.py, PASS. В processing.py
  два прежних F811 (sd_model/mask_blur), подтверждены на backup. Остальные
  старые замечания полного modules остаются;
  число зелёных тестов не является доказательством отсутствия других ошибок.
- Package verify PASS; check рабочего проекта: changes=0, UNKNOWN=0.
- 101 assets / 64 операции. Source/reference processing snapshot и контекстный
  patch согласованы, patch идемпотентен. Старый snapshot в files/modules/processing.py
  сначала дал FAIL нового cache-теста; сборщик исправлен, весь пакет проверен повторно.

Новые tests используют настоящие классы SD3/EmbeddingsWithFixes, оба этапа
аварийного lowvram setup, merge/repeat/disable LoRA, реальный API refresh,
реальные временные config files и 24 конкурирующих транзакции Options.
Никакие существующие assertions безопасности не ослаблялись. Старые заглушки
дополнены ensure_model_compatibility / set_and_save, AST-окружение — model_hijack.
I/O injection перенесён с прямого open на новую atomic-write границу; добавлены
реальные проверки fsync/replace failure, неизменности файла/памяти и retry.

Не проверены: конкретные Herrscher/IllustriousFlowmatch safetensors,
WAI-Illustrious-17 generation, CUDA/bfloat16/FP8 kernels, полный Gradio/auth/proxy,
межпроцессные config writers, произвольные внешние callbacks. Ничего не скачано
и не установлено, существующий Torch используется только с диска D:.
Ранее описанные v9 результаты ниже сохранены как история.

## Актуальные изменения v9

A1–A5 исправлены в networks.py, scripts/lora_script.py, новом lora_studio_access.py
и modules/textual_inversion/textual_inversion.py. Новый bias helper включён в
существующий AST harness без ослабления его assertions. Добавлены 9 CPU/AST
методов и 6 in-process FastAPI методов. До правок воспроизводители дали 5 runtime
FAIL и 1 API FAIL; после правок — 6 runtime/VAE PASS и 1 API PASS.

Runtime: 81 PASS; parser/schema/presets: 37 PASS; API: 6 PASS; Node: 17 PASS
с 51 UI assertion. Float8 проверен как CPU storage, не CUDA kernel. NaN/±Inf,
float64→float32 overflow, плохие размеры и частичная SDXL-пара отклоняются.
Ширины проверены и на нестандартных тестовых размерах, а не только 768/1280.
Bias: Linear/Conv2d, MHA и packed QKV; исходный None/обычный bias, fp16,
повтор, смена силы, несколько адаптеров, rollback/disable, отдельный Float8 bias test.

API: anonymous/wrong/malformed Basic, valid Basic, Gradio cookie, mixed mode,
forged/session-without-auth, no-auth и cross-origin write. Настоящий Gradio
не установлен и не запускался; pinned 3.41.2 auth/tokens contract сверён с
официальным исходником, в тестах app.auth/app.tokens подменены. Не заявлять
проверенную интеграцию произвольных версий Gradio или reverse proxy.

Backup: <HISTORICAL_AUDIT_ROOT>/before-20261105-a1-a5.
Полностью сохранены v8 и ZIP. UI, VAE implementation, defaults и lyco_helpers.py
не менялись. Ничего не устанавливалось. Независимый TI test обновлён для нового
раннего отказа: вместо prebuilt stub используется production import с реальными
encoder-width wrappers, сохраняется assertion об отсутствии регистрации.

Открытые пункты: реальная GPU/Gradio приёмка, U1/переделка UI, произвольные внешние
hooks, сбойная/конкурентная запись общего config. Аутентификация A1 не означает
полный security-аудит всего A1111. Исторические разделы ниже относятся к v8/v7.

## Перепроверка комплекта v9

Финальная проверка самого v9: 81 runtime PASS, 6 API PASS, 23 NumPy loading PASS,
установщик/inspector 25 PASS + 1 OS-symlink SKIP (включая synthetic install/rollback).
87 manifest assets, 60 операций; check проекта: Changes 0 / UNKNOWN 0. Ruff
изменённых Python PASS с project config. NumPy routing harness обновлён под
expected_shapes; добавлена проверка передачи значений, прежние assertions сохранены.
Независимый старый аудит v3 также снова 5 PASS.

## История v8

## Актуальное дополнение v8

Из runtime изменён modules/sd_vae.py. До исправления семь точек отказа standalone
VAE reload оставляли неполную модель доступной; теперь объект исключается из
active/cache, VAE marker очищается, исходное исключение сохраняется. Ошибка
resolve до изменения модели не удаляет исправный объект. Автоматического
восстановления прежних весов нет — требуется явная перезагрузка checkpoint.

Новый test_runtime_vae_failures.py: 3 метода, включая 7 mutation failure stages,
preflight, success/no-op и low-VRAM service-double ветку. Исходный runtime-набор:
72 PASS. Node 17 PASS, включая 51 DOM assertion. Никаких новых зависимостей.
Backup: <HISTORICAL_AUDIT_ROOT>/before-20261105-vae-ui.

В настоящем браузере на test/lora_studio_fixture.html: 55 PASS, включая четыре
режима контраста. Это mock-API стенд, не Gradio. Найдены UX-пробелы по сравнению
с исходными Weight Helper / Block Weight / LoRA Control: UX_AUDIT_RU.md.
Планы уточнены; UI-код, LoRA defaults и lyco_helpers.py не менялись.
Реальные CUDA/Gradio, backend hooks, projection output bias и custom migrations
остаются непроверенными. Предыдущие результаты ниже — история v7, не новое
заявление о повторном прохождении каждого старого теста.

## Перепроверка комплекта v8

Финальная перепроверка v8: из пакета 72 runtime PASS и 23 NumPy loading PASS;
установщик 25 PASS + 1 OS-symlink SKIP (26 всего), включая synthetic install/rollback.
Исходники: parser/schema/presets 37 PASS, независимые regression 5 PASS.
84 manifest assets, 59 операций; check рабочего проекта: Changes 0 / UNKNOWN 0.
Ruff изменённых Python PASS с конфигурацией проекта. Старые предупреждения
NumPy в D-venv и pytest base_url не устранены установкой зависимостей.
На статическом сервере отсутствовал SourceSansPro CSS; screenshot использует
fallback font. Точные размеры реального Gradio этим стендом не подтверждаются.

## История v7

## Актуальные изменения и результаты v7

Три runtime-файла: modules/sd_disable_initialization.py, modules/sd_models.py,
modules/textual_inversion/textual_inversion.py. Добавлен test_runtime_state_io.py
(13 тестов); AST-harnesses трёх старых наборов получили зависимости новых
выделенных helpers, без ослабления прежних assertions.

До исправлений новые тесты воспроизвели потерю metadata version, сбой dict
extra_state, превращение tensor extra_state в meta, игнорирование materialized
parameter dtype и повреждение прежних .pt/.optim при serialization failure.
Отдельно воспроизведены неверные cached FP8 decisions/global mode. При повторной
проверке собственного нового save-пути найден и закрыт legacy-sidecar fallback.

- Meta: сохраняются metadata и extra_state; dtype conversion применяется при
  материализации; проверены исключение first_stage_model и assign=True.
- TI: оба payload готовятся во временных файлах до замены; отдельный os.replace
  атомарен, пара — нет. Ошибка serialization сохраняет прежние файлы. Ошибка
  второго replace оставляет валидный новый .pt и устаревший .optim, который
  отвергается по SHA256. Маркер в новом .pt запрещает fallback на старый checksum-
  only .optim. Подлинная legacy-пара сохраняет прежнюю совместимость.
- FP8: actual target storage вместо глобального режима другой модели; точный
  checkpoint перечитывается при несовпадении настройки. Проверены четыре
  комбинации desired/stored, preflight failure и upcast реальных Float8 tensors
  перед загрузкой при global fp8=False.

Проверки рабочего проекта: 69 CPU/AST runtime PASS (56 прежних + 13 новых),
независимый аудит 5 PASS, parser/schema/presets 37 PASS, Node 17 PASS с 51 DOM
assertion, Ruff изменённых Python PASS. Torch 2.10.0+cpu/Python 3.12.10; прежние
предупреждения об отсутствии NumPy в D-venv и pytest base_url остаются.

Backup до этих изменений: <HISTORICAL_AUDIT_ROOT>/before-20261104-state-io.
v6 и его ZIP не изменялись. UI/LoRA defaults/lyco_helpers.py не менялись.
Ничего не скачивалось и не устанавливалось. План внутри проекта обновлён.

Ограничения: реальная GPU-генерация, браузер/Gradio, пользовательские модели,
training loop, внешние backend hooks и матрица версий не запускались. SHA256
связки optimizer — проверка соответствия, не подпись и не замена safe-unpickle.
Не гарантируются одновременная запись несколькими writers и power-loss durability
на всех файловых системах. Custom meta migrations до super, projection output
bias и самостоятельные ошибки VAE reload остаются отдельными проверками.

Повторный прогон из самого v7: 69 runtime PASS, 23 NumPy loading PASS,
установщик/инспектор 25 PASS и 1 SKIP (ОС не разрешает symlink) из 26.
Полный payload установлен и побайтово откачен в синтетической копии на D:.
82 manifest assets, 58 operations; check рабочего проекта: Changes 0, UNKNOWN 0.
Ruff PASS с конфигурацией pyproject.toml исходного проекта. Без неё пакетный
default lint отмечает три прежних E721, которые проект намеренно игнорирует.

---

# История: проверка кандидата v6 — 2026-11-04

## Изменения v6 и воспроизведение

До изменения runtime новый набор из 7 методов дал 7 failures и 2 errors с учётом
subtests. Причины сгруппированы в три исправления:

- modules/sd_models.py: не оставлять повреждённый cached checkpoint после ошибки
  transfer/VAE; checkpoint/hash settings обновляются только после успеха VAE.
  Остальные cached models сохраняются; автоматическое восстановление прежнего
  активного объекта не заявляется.
- extensions-builtin/Lora/networks.py: projected QkvLinear LoRA накапливает
  адаптеры в рабочем весе, использует fp16 cache и один финальный cast. Прежнее
  in-place Float8 + float падало. Отключение/ошибка восстанавливают storage backup.
- modules/sd_disable_initialization.py: rollback частичной установки, raw
  classmethod descriptors, owner check, независимая очистка и retry, сохранение
  primary error. Отдельные nested contexts поддерживаются; повторный вход того
  же активного экземпляра запрещён.

Набор test/test_runtime_edgecases.py расширен до 11 методов. В том числе
проверены все восемь позиций отказа установки meta-loader поверх LoraPatches,
restore failure и foreign owner, Class.method/Child.method, два LoRA с TE/DYN
в packed FP8, cache/no-cache, поздняя ошибка проекции и успешная повторная попытка.

## Результаты v6

- 56 CPU/production-AST runtime tests PASS: 21 stability + 12 recovery +
  12 compatibility + 11 edgecases. Запускались и на рабочем проекте, и из пакета.
- Прежний независимый reaudit_v3_20261103.py: 5 методов / 7 сценариев PASS.
- 23 NumPy LoRA loading tests PASS.
- 37 schema/parser/preset tests PASS; прежний pytest warning base_url остаётся.
- 17 Node tests PASS, в том числе 51 DOM assertion. CSS/layout в Node не проверены.
- Установщик/инспектор: 25 PASS, 1 SKIP (ОС не разрешает symlink) из 26.
  Весь payload установлен/побайтово восстановлен в синтетическом host на D:.
- Ruff новых/изменённых Python runtime/tests/build/install файлов PASS.
- verify/check: 81 assets, 58 operations; текущий проект: Changes 0, UNKNOWN 0.

Runtime-тесты используют настоящий Torch 2.10.0+cpu/Python 3.12.10, но выбирают
production definitions через AST и подменяют отсутствующие сервисы WebUI.
Предупреждение Torch об отсутствии NumPy в D-venv сохраняется; NumPy-тесты
выполнены отдельно имеющимся системным Python, ничего не устанавливалось.

## Исходники v6 и восстановление

Исходники уже изменены в рабочем проекте на D:. Снимок существовавших файлов:
<HISTORICAL_AUDIT_ROOT>/before-20261104-edgecases.
Это отдельный прямой backup; это НЕ ID журнала install.py.

v6 основан на v5. v5 и его ZIP сохранены. processing.py остаётся контекстным
patch из четырёх hunks, остальные runtime operations — известные полные файлы.
Пакет не универсален для произвольных форков. Не копируйте отдельные hooks
без согласованного modules/patches.py. lyco_helpers.py не менялся.

LORA_EVOLUTION_PLAN.md уточнён: runtime и настоящая приёмка обязательны перед
новой архитектурой. UI/default weights в этом шаге не менялись.

## Открытые проверки v6

На машине есть RTX 3050 Ti Laptop 4096 MiB (driver 566.07), но здесь используется
CPU Torch. В проверенном проекте нет checkpoint .ckpt/.safetensors; WebUI на
7860/7861/7862 не слушал. Диски вне проекта на модели не сканировались.

Не выполнены: настоящий SD1.5/SDXL/CEB/Velvette, CUDA forward/FP8, VRAM,
генерация с внешними attention extensions, реальные поля/цвета/адаптивность
Gradio UI, матрица Torch/transformers. SDXL TI training не поддерживается.

Отдельные открытые границы: arbitrary projection output bias; extra_state и
metadata в destructive meta loader; кеши при смене FP8 настройки; standalone
VAE reload failures; low-VRAM/backend lifecycle; atomic TI/optimizer save.
Не все эти риски воспроизведены; они не объявляются ни исправными, ни сломанными
без проверки. Неизвестные форматы нельзя принимать с молчаливой потерей данных.

v6 — кандидат с проверенными локальными исправлениями, не окончание всего плана
и не доказательство улучшения стиля/качества изображений.

---

# История: проверка v5 (состояние до v6)

## Что добавлено после v4

- modules/safe.py: разрешены точные ZIP-имена .format_version и
  .storage_alignment, записываемые установленным Torch. Не добавлены wildcard
  для произвольных файлов или fallback с отключением проверки pickle.
- modules/textual_inversion/textual_inversion.py: SDXL clip_l/clip_g отделены от
  метаданных; проверяются форма/число векторов; реализованы reloadable save и
  checksum обеих матриц. Формат и checksum обычных TI оставлены прежними.
- SDXL TI training не реализован. Старый single-encoder trainer теперь отказывает
  явно до изменения job state и подготовки dataset вместо позднего AttributeError.
- modules/devices.py: cleanup охватывает частичный отказ установки manual_cast,
  сохраняет наследование forward и не снимает чужой org_forward.
- Добавлен test/test_runtime_compatibility.py, 12 тестов.

## Воспроизведение до исправления

Новый набор дал 3 FAIL и 2 ERROR из 8 первоначальных тестов:
manual_cast оставлял hooks после отказа установки; safe-loader отвергал собственный
.pt из-за .format_version; SDXL TI падал на metadata/checksum и не проверял
совпадение количества векторов. Классический TI round-trip и delayed FP8/fp16-cache
прошли уже до исправлений. FP8 runtime в этом шаге не менялся.

## Повторные проверки после изменения

- 21 test_runtime_stability + 12 test_runtime_recovery + 12
  test_runtime_compatibility: 45 CPU/production-AST тестов PASS.
- Compatibility: сохранение/чтение обычного TI и optimizer state, SDXL metadata
  round-trip, checksum обоих encoder, mismatch counts, legacy Torch serialization,
  неизвестный ZIP member и безвредный запрещённый pickle global, ранний отказ
  SDXL training, реальное float8-хранилище с поздней передачей fp16 cache,
  manual_cast setup failure/наследование/вложенность и LoRA+cast в обоих порядках.
- Отдельный subprocess импортировал целиком текущий modules/safe.py с настоящими
  Torch 2.10.0+cpu и уже имевшимся NumPy 2.4.2. Глобально обёрнутый torch.load
  успешно прочёл self-saved tensor .pt. Подменены только config/reporting WebUI.
- 23 NumPy-теста LoRA загрузчика: PASS.
- 37 schema/parser/preset тестов и 17 Node-тестов (51 DOM-проверка): PASS.
  Прежнее предупреждение pytest об опции base_url сохраняется.
- Установщик/инспектор: 25 PASS, 1 SKIP из 26 (ОС не разрешает symlink).
  Весь payload установлен и побайтово восстановлен в синтетическом host на D:.
- Ruff изменённых runtime-файлов, теста и build/install скриптов: PASS.

## Источники и пакет

Рабочий проект на D: изменён напрямую. До редактирования сохранены три runtime-
файла и документация:
<HISTORICAL_AUDIT_ROOT>/before-20261104-compat

v5 содержит всё из v4, 80 проверяемых assets и 58 install operations.
Новая полная замена modules/safe.py разрешена установщиком только как конкретный
путь; неизвестные версии по-прежнему требуют проверки и явного opt-in.
processing.py сохраняет четыре контекстных фрагмента v4. v4 и его ZIP не менялись.
lyco_helpers.py не менялся. Библиотеки не устанавливались.

## Ограничения

Это не завершение всего плана и не гарантия безопасности произвольного pickle.
Неизвестные/недоверенные модели не следует загружать с отключённой защитой.

Torch проверен в одной CPU-среде (2.10.0, Python 3.12.10). Большинство тестов
изолирует production-функции через AST; не запускает сервер и настоящий checkpoint.
SDXL TI матрицы в тестах небольшие синтетические, не пользовательские файлы.
FP8 проверен как CPU-хранилище/merge/restore, не как CUDA inference или benchmark.
Optimizer state round-trip не означает поддержку обучения SDXL TI.

Остаются: реальные SD1.5/SDXL и CEB/Velvette, GPU generation, браузер,
разные версии библиотек, настоящие внешние attention extensions, сложные
cache-hit/VAE/backend failures и более широкое владение временными loader hooks.
Перенос идей из других WebUI остаётся после этих проверок.
